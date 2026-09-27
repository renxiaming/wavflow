from __future__ import annotations

import os
import random
from pathlib import Path

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi


DEFAULT_CAMPPLUS_ONNX = (
    "/home/work_nfs23/xmren/huawei_proj/huawei_streaming_cosyvoice/"
    "pretrain_models/CosyVoice2-0.5B/campplus.onnx"
)


def resolve_campplus_path(model_path: str | None = None) -> str:
    path = (
        model_path
        or os.environ.get("WAVTTS_SPK_EMBEDDING_MODEL")
        or os.environ.get("CAMPPLUS_ONNX")
    )
    if path is None:
        onnx_dir = os.environ.get("onnx_path")
        if onnx_dir:
            path = os.path.join(onnx_dir, "campplus.onnx")
    path = path or DEFAULT_CAMPPLUS_ONNX

    if os.path.isdir(path):
        path = os.path.join(path, "campplus.onnx")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "campplus.onnx not found. Set model.speaker_embedding.model_path or "
            "WAVTTS_SPK_EMBEDDING_MODEL to a valid file."
        )
    return path


class SpeakerEmbeddingExtractor:
    """CosyVoice2-compatible CAM++ speaker embedding extractor."""

    def __init__(
        self,
        model_path: str | None = None,
        sample_rate: int = 16000,
        max_seconds: float = 10.0,
        providers: list[str] | None = None,
    ):
        try:
            import onnxruntime
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "onnxruntime is required for online speaker embedding extraction. "
                "Install onnxruntime or onnxruntime-gpu in the training environment."
            ) from exc

        self.model_path = resolve_campplus_path(model_path)
        self.sample_rate = int(sample_rate)
        self.max_len = int(max_seconds * self.sample_rate)

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            self.model_path,
            sess_options=option,
            providers=providers or ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def _prepare_speech(self, speech: torch.Tensor, length: int | None = None) -> torch.Tensor:
        speech = speech.detach().to(device="cpu", dtype=torch.float32)
        if speech.ndim == 2:
            speech = speech.mean(dim=0)
        if length is not None:
            speech = speech[: int(length)]
        speech = speech.reshape(1, -1)

        if speech.shape[1] > self.max_len:
            start = random.randint(0, speech.shape[1] - self.max_len)
            speech = speech[:, start : start + self.max_len]
        return speech

    @torch.no_grad()
    def extract(self, speech: torch.Tensor, length: int | None = None, device=None) -> torch.Tensor:
        speech = self._prepare_speech(speech, length=length)
        feat = kaldi.fbank(
            speech,
            num_mel_bins=80,
            dither=0,
            sample_frequency=self.sample_rate,
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.session.run(
            None,
            {self.input_name: feat.unsqueeze(0).numpy()},
        )[0].flatten()
        out = torch.from_numpy(embedding).float()
        return out.to(device=device) if device is not None else out

    @torch.no_grad()
    def extract_batch(
        self,
        wav: torch.Tensor,
        lengths: torch.Tensor | None = None,
        device=None,
    ) -> torch.Tensor:
        if wav.ndim != 2:
            raise ValueError(f"Expected padded waveform [B, N], got {tuple(wav.shape)}")
        device = wav.device if device is None else device
        embeddings = []
        for i in range(wav.shape[0]):
            length = int(lengths[i].item()) if lengths is not None else None
            embeddings.append(self.extract(wav[i], length=length, device=device))
        return torch.stack(embeddings, dim=0)

    @torch.no_grad()
    def extract_from_path(self, wav_path: str | Path, device=None) -> torch.Tensor:
        speech, sr = torchaudio.load(str(wav_path))
        if speech.shape[0] > 1:
            speech = speech.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            speech = torchaudio.functional.resample(speech, sr, self.sample_rate)
        return self.extract(speech, device=device)
