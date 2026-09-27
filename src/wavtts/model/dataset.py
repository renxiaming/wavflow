import io
import json
import random
from importlib.resources import files

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from datasets import Dataset as Dataset_
from datasets import load_from_disk
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from wavtts.model.lance.aslp.tools.lance_pack import LanceReader
from wavtts.model.lance.serialize import TorchShmSerializedList, get_rank
from wavtts.model.lance.text_index import EmiliaTextIndex
from wavtts.model.lance.utils import load_co, load_li
from wavtts.model.utils import convert_char_to_pinyin

try:
    torch.multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass
torch.multiprocessing.current_process().authkey = b"my_authkey"



class CustomDataset(Dataset):
    def __init__(
        self,
        custom_dataset: Dataset,
        durations=None,
        target_sample_rate=16_000,
        wav_frame_len: int = 160,
        **_,
    ):
        self.data = custom_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.wav_frame_len = wav_frame_len

        self._resamplers = {}

    def get_frame_len(self, index):
        if (
            self.durations is not None
        ):  # Please make sure the separately provided durations are correct, otherwise 99.99% OOM
            return self.durations[index] * self.target_sample_rate / self.wav_frame_len
        return self.data[index]["duration"] * self.target_sample_rate / self.wav_frame_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        while True:
            row = self.data[index]
            audio_path = row["audio_path"]
            text = row["text"]
            duration = row["duration"]

            # filter by given length
            if 0.3 <= duration <= 30:
                break  # valid

            index = (index + 1) % len(self.data)

        audio, source_sample_rate = torchaudio.load(audio_path)

        # make sure mono input
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # resample if necessary
        if source_sample_rate != self.target_sample_rate:
            if source_sample_rate not in self._resamplers:
                self._resamplers[source_sample_rate] = torchaudio.transforms.Resample(
                    source_sample_rate, self.target_sample_rate
                )
            audio = self._resamplers[source_sample_rate](audio)

        return {
            "wav": audio.squeeze(0),
            "text": text,
        }


class EmiliaLanceDataset(Dataset):
    """Emilia dataset backed by Lance wav index (.co/.li) and sqlite/jsonl text lookup."""

    def __init__(
        self,
        file_lst,
        lance_connections,
        text_index: EmiliaTextIndex | None,
        tokenizer: str = "pinyin",
        polyphone: bool = True,
        target_sample_rate=16_000,
        wav_frame_len: int = 160,
        min_duration: float = 0.3,
        max_duration: float = 30.0,
        load_speech_tokens: bool = False,
        **_,
    ):
        self.file_lst = file_lst
        if lance_connections and isinstance(lance_connections[0], tuple):
            self.lance_connections = [
                LanceReader(path, target_cls=cls) for path, cls in lance_connections
            ]
        else:
            self.lance_connections = lance_connections
        self.text_index = text_index
        self.tokenizer = tokenizer
        self.polyphone = polyphone
        self.target_sample_rate = target_sample_rate
        self.wav_frame_len = wav_frame_len
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.load_speech_tokens = load_speech_tokens
        self._resamplers = {}
        self.len = len(file_lst)

    @staticmethod
    def init_data(lance_path: str, shuffle_seed: int = 42):
        conns = load_co(lance_path + ".co")
        if get_rank() > 0:
            return TorchShmSerializedList([]), conns

        lance_file_lst = load_li(lance_path + ".li")
        random.seed(shuffle_seed)
        random.shuffle(lance_file_lst)
        return TorchShmSerializedList(lance_file_lst), conns

    def _prepare_text(self, utt: str, raw_text: str) -> list[str]:
        from wavtts.model.lance.text_prep import prepare_pinyin_text

        # "custom" only selects vocab.txt path; text tokens are still pinyin syllables.
        if self.tokenizer in ("pinyin", "custom"):
            return prepare_pinyin_text(raw_text, polyphone=self.polyphone)
        return list(raw_text)

    def get_frame_len(self, index):
        item = self.file_lst[index].strip()
        parts = item.split("|")
        if len(parts) >= 6:
            token_length = int(parts[-1])
            duration = token_length / 25.0
        else:
            duration = 10.0
        return duration * self.target_sample_rate / self.wav_frame_len

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        for _ in range(3):
            item = self.file_lst[index].strip()
            parts = item.split("|")
            if len(parts) < 6:
                index = (index + 1) % self.len
                continue

            utt, wav_rowid, wav_offset, token_rowid, token_offset, token_length = (
                parts[0],
                int(parts[1]),
                int(parts[2]),
                int(parts[3]),
                int(parts[4]),
                int(parts[5]),
            )
            duration = token_length / 25.0
            if not (self.min_duration <= duration <= self.max_duration):
                index = (index + 1) % self.len
                continue

            raw_text = self.text_index.get_text(utt) if self.text_index is not None else None
            if raw_text is None and not self.load_speech_tokens:
                index = (index + 1) % self.len
                continue

            try:
                wav_data = self.lance_connections[wav_offset].get_datas_by_rowids([wav_rowid])[0]
                audio, source_sample_rate = torchaudio.load(io.BytesIO(wav_data.mp3_binary))
                speech_tokens = None
                if self.load_speech_tokens:
                    token_data = self.lance_connections[token_offset].get_datas_by_rowids([token_rowid])[0].data
                    if isinstance(token_data, np.ndarray):
                        speech_tokens = torch.from_numpy(token_data).long()
                    else:
                        speech_tokens = torch.tensor(token_data, dtype=torch.long)
                    speech_tokens = speech_tokens.reshape(-1)
            except Exception:
                index = (index + 1) % self.len
                continue

            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            if source_sample_rate != self.target_sample_rate:
                if source_sample_rate not in self._resamplers:
                    self._resamplers[source_sample_rate] = torchaudio.transforms.Resample(
                        source_sample_rate, self.target_sample_rate
                    )
                audio = self._resamplers[source_sample_rate](audio)

            result = {
                "wav": audio.squeeze(0),
                "text": self._prepare_text(utt, raw_text) if raw_text is not None else ["a"],
            }
            if self.load_speech_tokens:
                result["speech_tokens"] = speech_tokens
            return result

        result = {"wav": torch.zeros(self.target_sample_rate), "text": ["a"]}
        if self.load_speech_tokens:
            result["speech_tokens"] = torch.zeros(1, dtype=torch.long)
        return result


# Dynamic Batch Sampler
class DynamicBatchSampler(Sampler[list[int]]):
    """Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    3.  Shuffle batches each epoch while maintaining reproducibility.
    """

    def __init__(
        self, sampler: Sampler[int], frames_threshold: int, max_samples=0, random_seed=None, drop_residual: bool = False
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.epoch = 0

        indices, batches = [], []
        data_source = self.sampler.data_source

        for idx in tqdm(
            self.sampler, desc="Sorting with sampler... if slow, check whether dataset is provided with duration"
        ):
            indices.append((idx, data_source.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        for idx, frame_len in tqdm(
            indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu"
        ):
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0

        if not drop_residual and len(batch) > 0:
            batches.append(batch)

        del indices
        self.batches = batches

        # Ensure even batches with accelerate BatchSamplerShard cls under frame_per_batch setting
        self.drop_last = True

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler."""
        self.epoch = epoch

    def __iter__(self):
        # Use both random_seed and epoch for deterministic but different shuffling per epoch
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            # Use PyTorch's random permutation for better reproducibility across PyTorch versions
            indices = torch.randperm(len(self.batches), generator=g).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches
        return iter(batches)

    def __len__(self):
        return len(self.batches)


# Load dataset


def load_dataset(
    dataset_name: str,
    tokenizer: str = "pinyin",
    dataset_type: str = "CustomDataset",
    audio_type: str = "raw",
    waveform_kwargs: dict = dict(),
    lance_path: str | None = None,
    text_index_path: str | None = None,
    text_jsonl_path: str | None = None,
    polyphone: bool = True,
    load_speech_tokens: bool = False,
) -> Dataset:
    """
    WavTTS only supports raw waveform datasets.
    dataset_type:
      - "CustomDataset": use tokenizer name and default data path
      - "CustomDatasetPath": pass the full path to a prepared dataset
      - "EmiliaLance": read wav from Lance (.co/.li) and text from sqlite/jsonl
    """

    print("Loading dataset ...")

    if audio_type != "raw":
        raise ValueError("WavTTS only supports raw waveform datasets; audio_type must be 'raw'.")

    if dataset_type == "EmiliaLance":
        if not lance_path:
            raise ValueError("EmiliaLance requires datasets.lance_path")
        print(f"Using Emilia Lance data: {lance_path}")
        file_lst, lance_connections = EmiliaLanceDataset.init_data(lance_path)
        if text_index_path or text_jsonl_path or not load_speech_tokens:
            text_index = EmiliaTextIndex(index_path=text_index_path, jsonl_path=text_jsonl_path)
        else:
            text_index = None
        return EmiliaLanceDataset(
            file_lst=file_lst,
            lance_connections=lance_connections,
            text_index=text_index,
            tokenizer=tokenizer,
            polyphone=polyphone,
            load_speech_tokens=load_speech_tokens,
            **waveform_kwargs,
        )

    if dataset_type == "CustomDataset":
        rel_data_path = str(files("wavtts").joinpath(f"../../data/{dataset_name}_{tokenizer}"))
        try:
            train_dataset = load_from_disk(f"{rel_data_path}/raw")
        except:  # noqa: E722
            train_dataset = Dataset_.from_file(f"{rel_data_path}/raw.arrow")
        with open(f"{rel_data_path}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(train_dataset, durations=durations, **waveform_kwargs)

    elif dataset_type == "CustomDatasetPath":
        try:
            train_dataset = load_from_disk(f"{dataset_name}/raw")
        except:  # noqa: E722
            train_dataset = Dataset_.from_file(f"{dataset_name}/raw.arrow")

        with open(f"{dataset_name}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(train_dataset, durations=durations, **waveform_kwargs)

    else:
        raise ValueError(f"Unsupported dataset_type for WavTTS wav-only training: {dataset_type}")

    return train_dataset


# collation


def collate_fn(batch):
    text = [item["text"] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])
    
    wavs = [item["wav"] for item in batch]
    has_wav = wavs[0] is not None
    if has_wav:
        wav_lengths = torch.LongTensor([w.shape[0] for w in wavs])
        max_wav_len = wav_lengths.max().item()

        padded_wavs = []
        for w in wavs:
            pad_len = max_wav_len - w.shape[0]
            padded_wavs.append(
                F.pad(w, (0, pad_len), value=0.0)
            )

        wavs = torch.stack(padded_wavs)   # [B, T_wav]
    else:
        wavs = None
        wav_lengths = None
    

    result = dict(
        text=text,
        text_lengths=text_lengths,
        wav=wavs,
        wav_lengths=wav_lengths,
    )

    if "speech_tokens" in batch[0]:
        speech_tokens = [item["speech_tokens"].long().reshape(-1) for item in batch]
        result["speech_tokens"] = torch.nn.utils.rnn.pad_sequence(
            speech_tokens, batch_first=True, padding_value=-1
        )
        result["speech_token_lengths"] = torch.LongTensor([t.numel() for t in speech_tokens])

    return result
