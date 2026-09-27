from __future__ import annotations

import argparse
import json
import os
import string
import sys
from pathlib import Path


PROTOCOL_STDOUT = sys.stdout
sys.stdout = sys.stderr


def emit(payload: dict):
    PROTOCOL_STDOUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
    PROTOCOL_STDOUT.flush()


def normalize_device(device: str) -> str:
    value = str(device)
    if value.startswith("cuda:"):
        index = value.split(":", 1)[1]
    elif value.isdigit():
        index = value
    else:
        return value

    # The training shell may expose physical GPUs as "1" or "2,3".
    # Convert the requested physical id to the worker-local CUDA index.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        visible_ids = [item.strip() for item in visible.split(",") if item.strip()]
        if index in visible_ids:
            index = str(visible_ids.index(index))
    return f"cuda:{index}"


def load_sim_model(sim_root: Path, device: str):
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ.setdefault(
        "SIM_WER_WAVLM_BASE_CKPT",
        "/home/A02_tmpdata1/wenhaoli/code/seed_eval/ckpts/wavlm_large.pt",
    )
    if str(sim_root) not in sys.path:
        sys.path.insert(0, str(sim_root))
    os.chdir(sim_root)

    import torch
    from models.ecapa_tdnn import ECAPA_TDNN_SMALL

    checkpoint = os.environ.get(
        "SIM_WER_WAVLM_CKPT",
        str(sim_root / "ckpt" / "wavlm_large_finetune.pth"),
    )
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"WavLM speaker checkpoint not found: {checkpoint}")
    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    state = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    model.load_state_dict(state["model"], strict=False)
    model.eval().to(device)
    return model


def sim_embedding(model, wav_path: str, device: str, sample_rate: int = 16000):
    import soundfile as sf
    import torch
    from torchaudio.transforms import Resample

    wav, sr = sf.read(wav_path)
    if len(wav) == 0:
        return None
    wav = torch.from_numpy(wav).unsqueeze(0).float().to(device)
    if len(wav.shape) == 3:
        wav = wav[:, :, 0]
    if sr != sample_rate:
        wav = Resample(orig_freq=sr, new_freq=sample_rate).to(device)(wav)
    with torch.no_grad():
        return model(wav)


def score_sim(model, items: list[dict], device: str) -> dict[str, float]:
    import torch.nn.functional as F

    values = {}
    for item in items:
        sample_id = item["sample_id"]
        prompt_emb = sim_embedding(model, item["prompt_path"], device=device)
        generated_emb = sim_embedding(model, item["generated_path"], device=device)
        if prompt_emb is None or generated_emb is None:
            values[sample_id] = 0.0
        else:
            values[sample_id] = float(F.cosine_similarity(prompt_emb, generated_emb).item())
    return values


def load_wer_model(sim_root: Path, language: str, device: str):
    if str(sim_root) not in sys.path:
        sys.path.insert(0, str(sim_root))
    os.chdir(sim_root)

    if language == "zh":
        from funasr import AutoModel

        root = sim_root / "ckpt" / "funasr"
        return AutoModel(
            model=os.environ.get(
                "ZH_ASR_MODEL",
                str(root / "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"),
            ),
            vad_model=os.environ.get(
                "ZH_ASR_VAD_MODEL",
                str(root / "speech_fsmn_vad_zh-cn-16k-common-pytorch"),
            ),
            punc_model=os.environ.get(
                "ZH_ASR_PUNC_MODEL",
                str(root / "punc_ct-transformer_cn-en-common-vocab471067-large"),
            ),
            model_hub="ms",
            disable_pbar=True,
            device=device,
        )

    if language == "en":
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        model_id = os.environ.get("EN_ASR_MODEL_ID", str(sim_root / "ckpt" / "whisper-large-v3"))
        processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device).eval()
        return processor, model

    raise ValueError(f"Unsupported language: {language}")


def transcribe(model_bundle, wav_path: str, language: str, device: str) -> str:
    import scipy.signal
    import soundfile as sf
    import torch

    if language == "zh":
        import zhconv

        try:
            result = model_bundle.generate(input=wav_path, batch_size_s=300)
            text = result[0]["text"] if result else ""
            return zhconv.convert(text, "zh-cn")
        except Exception as exc:
            print(f"[WARN] ASR failed for {wav_path}: {exc}", file=sys.stderr)
            return ""

    processor, model = model_bundle
    wav, sample_rate = sf.read(wav_path)
    if sample_rate != 16000:
        wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sample_rate))
    features = processor(wav, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    forced_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
    with torch.no_grad():
        predicted_ids = model.generate(features, forced_decoder_ids=forced_ids)
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


def wer_from_text(hypothesis: str, reference: str, language: str) -> float:
    from jiwer import compute_measures
    from zhon.hanzi import punctuation

    punctuation_all = punctuation + string.punctuation + "-"
    for mark in punctuation_all:
        if mark == "'":
            continue
        reference = reference.replace(mark, "")
        hypothesis = hypothesis.replace(mark, "")
    reference = reference.replace("-", " ").replace("  ", " ")
    hypothesis = hypothesis.replace("-", " ").replace("  ", " ")
    if language == "zh":
        reference = " ".join(reference)
        hypothesis = " ".join(hypothesis)
    else:
        reference = reference.lower()
        hypothesis = hypothesis.lower()
    return float(compute_measures(reference, hypothesis)["wer"])


def score_wer(model_bundle, items: list[dict], language: str, device: str) -> dict[str, dict]:
    values = {}
    for item in items:
        sample_id = item["sample_id"]
        hypothesis = transcribe(model_bundle, item["generated_path"], language, device)
        values[sample_id] = {
            "wer": wer_from_text(hypothesis, item["target_text"], language),
            "hypothesis": hypothesis,
        }
    return values


def main():
    parser = argparse.ArgumentParser(description="Resident WER/SIM reward worker.")
    parser.add_argument("--mode", choices=["sim", "wer"], required=True)
    parser.add_argument("--language", choices=["zh", "en"], default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sim-root", default="third_party/sim_wer_test")
    args = parser.parse_args()

    device = normalize_device(args.device)
    sim_root = Path(args.sim_root).expanduser().resolve()

    if args.mode == "sim":
        model = load_sim_model(sim_root, device)
    else:
        if args.language is None:
            raise ValueError("--language is required when --mode=wer")
        model = load_wer_model(sim_root, args.language, device)

    emit({"status": "ready", "mode": args.mode, "language": args.language, "device": device})

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            request = json.loads(raw_line)
            if request.get("op") == "shutdown":
                emit({"id": request.get("id"), "ok": True, "status": "bye"})
                break
            items = request.get("items", [])
            if args.mode == "sim":
                values = score_sim(model, items, device)
            else:
                values = score_wer(model, items, args.language, device)
            emit({"id": request.get("id"), "ok": True, "values": values})
        except Exception as exc:
            emit({"id": request.get("id") if "request" in locals() else None, "ok": False, "error": repr(exc)})


if __name__ == "__main__":
    main()
