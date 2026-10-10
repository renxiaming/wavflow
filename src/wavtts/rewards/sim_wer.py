from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import socket
import string
import subprocess
import sys
import tempfile
from pathlib import Path

import scipy.signal
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio


class _ResidentWorker:
    def __init__(
        self,
        *,
        mode: str,
        language: str | None,
        python_executable: str,
        worker_script: Path,
        sim_root: Path,
        device: str,
    ):
        self.mode = mode
        self.language = language
        self.python_executable = python_executable
        self.worker_script = worker_script
        self.sim_root = sim_root
        self.device = device
        self.process: subprocess.Popen[str] | None = None
        self.next_id = 0
        self._start()

    def _start(self):
        command = [
            self.python_executable,
            str(self.worker_script),
            "--mode",
            self.mode,
            "--device",
            str(self.device),
            "--sim-root",
            str(self.sim_root),
        ]
        if self.language is not None:
            command.extend(["--language", self.language])
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            cwd=str(self.sim_root),
            text=True,
            bufsize=1,
            env=env,
        )
        ready = self._read()
        if ready.get("status") != "ready":
            self.close()
            raise RuntimeError(f"Resident {self.mode} worker failed to start: {ready}")

    def _read(self) -> dict:
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("Resident reward worker is not running.")
        line = self.process.stdout.readline()
        if not line:
            return {
                "status": "dead",
                "returncode": self.process.poll(),
            }
        return json.loads(line)

    def request(self, items: list[dict]) -> dict:
        import json

        if self.process is None or self.process.stdin is None:
            raise RuntimeError("Resident reward worker is not running.")
        request_id = self.next_id
        self.next_id += 1
        self.process.stdin.write(json.dumps({"id": request_id, "items": items}, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        response = self._read()
        if not response.get("ok"):
            raise RuntimeError(
                f"Resident {self.mode} worker request failed: {response.get('error', response)}"
            )
        return response.get("values", {})

    def close(self):
        process = self.process
        self.process = None
        if process is None:
            return
        try:
            if process.stdin is not None and process.poll() is None:
                process.stdin.write(json.dumps({"id": -1, "op": "shutdown"}) + "\n")
                process.stdin.flush()
                process.stdin.close()
            process.wait(timeout=10)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
            process.kill()
            process.wait()


class _RewardServerClient:
    def __init__(self, host: str, port: int, timeout: float = 3600.0):
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)

    def request(
        self,
        *,
        generated_paths: list[str],
        prompt_paths: list[str],
        target_texts: list[str],
        languages: list[str | None],
        reference_paths: list[str | None] | None = None,
    ) -> dict[str, list[float]]:
        payload = {
            "op": "score",
            "generated_paths": [str(Path(path).resolve()) for path in generated_paths],
            "prompt_paths": [str(Path(path).resolve()) for path in prompt_paths],
            "target_texts": target_texts,
            "languages": languages,
            "reference_paths": reference_paths,
        }
        with socket.create_connection((self.host, self.port), timeout=10.0) as connection:
            connection.settimeout(self.timeout)
            connection.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
            stream = connection.makefile("r", encoding="utf-8")
            response = json.loads(stream.readline())
        if not response.get("ok"):
            raise RuntimeError(f"Reward server request failed: {response.get('error', response)}")
        return response["result"]


class SimWerReward:
    """ASR accuracy, speaker similarity and perceptual MOS rewards."""

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        sim_device: str | None = None,
        wer_device: str | None = None,
        mos_device: str | None = None,
        sim_root: str = (
            "/home/A02_tmpdata0/xmren/code/grpo_flow/wavtts3.1_grpo/"
            "portable_wavtts_grpo/third_party/sim_wer_test"
        ),
        wer_weight: float = 0.2,
        sim_weight: float = 0.8,
        mos_weight: float = 0.0,
        language: str = "auto",
        backend: str = "resident",
        wer_script: str | None = None,
        sim_script: str | None = None,
        work_dir: str | None = None,
        keep_work_dir: bool = False,
        sim_python: str | None = None,
        wer_python: str | None = None,
        mos_python: str | None = None,
        resident_worker: str | None = None,
        server_host: str = "127.0.0.1",
        server_port: int = 29631,
        server_timeout: float = 3600.0,
    ):
        self.device = device
        self.sim_device = sim_device or device
        self.wer_device = wer_device or device
        self.mos_device = mos_device or device
        self.sim_root = Path(sim_root)
        self.wer_weight = float(wer_weight)
        self.sim_weight = float(sim_weight)
        self.mos_weight = float(mos_weight)
        if self.wer_weight < 0 or self.sim_weight < 0 or self.mos_weight < 0:
            raise ValueError("Reward weights must be non-negative.")
        if self.wer_weight == 0 and self.sim_weight == 0 and self.mos_weight == 0:
            raise ValueError("At least one reward weight must be positive.")
        self.language = language
        self.backend = backend
        self.wer_script = Path(wer_script or self.sim_root / "run_wer.sh")
        self.sim_script = Path(sim_script or self.sim_root / "run_sim.sh")
        self.work_dir = Path(work_dir).expanduser() if work_dir else None
        if self.work_dir is not None:
            self.work_dir.mkdir(parents=True, exist_ok=True)
        self.keep_work_dir = keep_work_dir
        self.sim_python = sim_python or sys.executable
        self.wer_python = wer_python or sys.executable
        self.mos_python = mos_python or sys.executable
        self.resident_worker = Path(resident_worker or Path(__file__).with_name("resident_reward_worker.py"))
        self._resident_workers: dict[tuple[str, str | None], _ResidentWorker] = {}
        self._server_client = _RewardServerClient(server_host, server_port, server_timeout)
        atexit.register(self.close)
        self._sim_model = None
        self._zh_asr = None
        self._en_asr = None

    def _load_sim(self):
        if self._sim_model is not None:
            return
        root = str(self.sim_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        from models.ecapa_tdnn import ECAPA_TDNN_SMALL

        checkpoint = os.environ.get(
            "SIM_WER_WAVLM_CKPT",
            str(self.sim_root / "ckpt" / "wavlm_large_finetune.pth"),
        )
        model = ECAPA_TDNN_SMALL(
            feat_dim=1024,
            feat_type="wavlm_large",
            config_path=None,
        )
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model"], strict=False)
        self._sim_model = model.eval().to(self.device)

    def _sim_embedding(self, path: str) -> torch.Tensor:
        self._load_sim()
        wav, sample_rate = sf.read(path)
        if len(wav) == 0:
            return torch.zeros(1, 192, device=self.device)
        wav = torch.from_numpy(wav).float()
        if wav.ndim > 1:
            wav = wav[:, 0]
        wav = wav.unsqueeze(0).to(self.device)
        if sample_rate != 16000:
            wav = torchaudio.functional.resample(wav, sample_rate, 16000)
        with torch.no_grad():
            return self._sim_model(wav)

    def _load_wer(self, language: str):
        if language == "zh" and self._zh_asr is None:
            from funasr import AutoModel

            root = self.sim_root / "ckpt" / "funasr"
            self._zh_asr = AutoModel(
                model=str(root / "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"),
                vad_model=str(root / "speech_fsmn_vad_zh-cn-16k-common-pytorch"),
                punc_model=str(root / "punc_ct-transformer_cn-en-common-vocab471067-large"),
                model_hub="ms",
                disable_pbar=True,
                device=self.device,
            )
        elif language == "en" and self._en_asr is None:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            model_id = os.environ.get(
                "EN_ASR_MODEL_ID",
                str(self.sim_root / "ckpt" / "whisper-large-v3"),
            )
            processor = WhisperProcessor.from_pretrained(model_id)
            model = WhisperForConditionalGeneration.from_pretrained(model_id).to(self.device).eval()
            self._en_asr = (processor, model)

    @staticmethod
    def _is_chinese(text: str) -> bool:
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    def _resolve_language(self, text: str, language: str | None) -> str:
        language = language or self.language
        if language in {"zh", "en"}:
            return language
        return "zh" if self._is_chinese(text) else "en"

    def _transcribe(self, path: str, language: str) -> str:
        self._load_wer(language)
        if language == "zh":
            import zhconv

            try:
                result = self._zh_asr.generate(input=path, batch_size_s=300)
                text = result[0]["text"] if result else ""
                return zhconv.convert(text, "zh-cn")
            except Exception:
                return ""

        processor, model = self._en_asr
        wav, sample_rate = sf.read(path)
        if sample_rate != 16000:
            wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sample_rate))
        features = processor(wav, sampling_rate=16000, return_tensors="pt").input_features
        features = features.to(self.device)
        forced_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
        with torch.no_grad():
            predicted_ids = model.generate(features, forced_decoder_ids=forced_ids)
        return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

    @staticmethod
    def _wer(hypothesis: str, reference: str, language: str) -> float:
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

    @staticmethod
    def _safe_id(value: str, index: int) -> str:
        clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
        return clean or f"sample_{index:06d}"

    @staticmethod
    def _gpu_id(device: str) -> str:
        match = re.search(r"(\d+)$", str(device))
        return match.group(1) if match else "0"

    @staticmethod
    def _result_from_metrics(
        wer_values: list[float],
        sim_values: list[float],
        mos_values: list[float] | None = None,
        *,
        wer_weight: float,
        sim_weight: float,
        mos_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        wer_tensor = torch.tensor(wer_values, dtype=torch.float32)
        sim_tensor = torch.tensor(sim_values, dtype=torch.float32)
        mos_tensor = torch.tensor(
            mos_values if mos_values is not None else [0.0] * len(wer_values),
            dtype=torch.float32,
        )
        # FlowTTS-GRPO uses ASR accuracy (1-CER/WER), mapped cosine SIM and
        # the raw MOS predictor output. Cross-objective normalization is done
        # over the complete distributed rollout batch in train_grpo.py.
        wer_reward = (1.0 - wer_tensor).clamp(min=0.0, max=1.0)
        sim_reward = ((sim_tensor + 1.0) * 0.5).clamp(min=0.0, max=1.0)
        mos_reward = mos_tensor
        reward = wer_weight * wer_reward + sim_weight * sim_reward + mos_weight * mos_reward
        return {
            "reward": reward,
            "wer": wer_tensor,
            "sim": sim_tensor,
            "mos": mos_tensor,
            "wer_reward": wer_reward,
            "sim_reward": sim_reward,
            "mos_reward": mos_reward,
        }

    def _score_mos_resident(self, generated_paths: list[str], sample_ids: list[str]) -> dict[str, float]:
        if self.mos_weight <= 0:
            return {sample_id: 0.0 for sample_id in sample_ids}
        items = [
            {"sample_id": sample_id, "generated_path": str(Path(path).resolve())}
            for sample_id, path in zip(sample_ids, generated_paths)
        ]
        return self._get_resident_worker("mos").request(items)

    def _score_with_scripts(
        self,
        *,
        generated_paths: list[str],
        prompt_paths: list[str],
        target_texts: list[str],
        languages: list[str | None],
        reference_paths: list[str | None],
    ) -> dict[str, torch.Tensor]:
        if self.wer_weight > 0 and not self.wer_script.exists():
            raise FileNotFoundError(f"WER script not found: {self.wer_script}")
        if self.sim_weight > 0 and not self.sim_script.exists():
            raise FileNotFoundError(f"SIM script not found: {self.sim_script}")

        resolved_languages = [
            self._resolve_language(text, language)
            for text, language in zip(target_texts, languages)
        ]
        effective_target_texts = list(target_texts)
        root = Path(
            tempfile.mkdtemp(
                prefix="wavtts_grpo_reward_",
                dir=str(self.work_dir) if self.work_dir else None,
            )
        )
        generated_dir = root / "generated"
        sim_output_dir = root / "sim"
        wer_output_dir = root / "wer"
        generated_dir.mkdir()
        sim_output_dir.mkdir()
        wer_output_dir.mkdir()

        try:
            safe_ids = [
                f"{self._safe_id(Path(path).stem, index)}_{index:06d}"
                for index, path in enumerate(generated_paths)
            ]
            for safe_id, source in zip(safe_ids, generated_paths):
                shutil.copy2(source, generated_dir / f"{safe_id}.wav")

            all_meta = root / "meta_all.lst"
            with all_meta.open("w", encoding="utf-8") as handle:
                for safe_id, prompt, text in zip(safe_ids, prompt_paths, effective_target_texts):
                    handle.write(f"{safe_id}||{Path(prompt).resolve()}|{text}\n")

            sim_values_by_id: dict[str, float] = {safe_id: 0.0 for safe_id in safe_ids}
            if self.sim_weight > 0:
                subprocess.run(
                    [
                        "bash",
                        str(self.sim_script),
                        str(generated_dir),
                        str(all_meta),
                        str(sim_output_dir),
                        str(self.device),
                    ],
                    check=True,
                )

                sim_values_by_id = {}
                sim_file = sim_output_dir / "total_similarity.txt"
                if not sim_file.exists():
                    raise RuntimeError(f"SIM script did not write {sim_file}")
                for line in sim_file.read_text(encoding="utf-8").splitlines():
                    match = re.match(r"^([^:]+):\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*$", line)
                    if match:
                        sim_values_by_id[match.group(1).strip()] = float(match.group(2))

                missing_sim = [safe_id for safe_id in safe_ids if safe_id not in sim_values_by_id]
                if missing_sim:
                    raise RuntimeError(f"SIM results are missing samples: {missing_sim}")

            wer_values_by_id: dict[str, float] = {safe_id: 0.0 for safe_id in safe_ids}
            ids_by_stem = {f"{safe_id}.wav": safe_id for safe_id in safe_ids}
            for language in sorted(set(resolved_languages)) if self.wer_weight > 0 else []:
                indices = [
                    index
                    for index, item_language in enumerate(resolved_languages)
                    if item_language == language
                ]
                meta_path = root / f"meta_{language}.lst"
                with meta_path.open("w", encoding="utf-8") as handle:
                    for index in indices:
                        handle.write(
                            f"{safe_ids[index]}||{Path(prompt_paths[index]).resolve()}|"
                            f"{effective_target_texts[index]}\n"
                        )
                language_output = wer_output_dir / language
                language_output.mkdir()
                subprocess.run(
                    [
                        "bash",
                        str(self.wer_script),
                        str(meta_path),
                        language,
                        f"grpo_{os.getpid()}_{language}",
                        "1",
                        self._gpu_id(self.device),
                        str(generated_dir),
                        str(language_output),
                    ],
                    check=True,
                )
                result_files = sorted(language_output.glob("thread_metas_*/results/thread-*.wer.out"))
                for result_file in result_files:
                    for line in result_file.read_text(encoding="utf-8").splitlines():
                        fields = line.split("\t")
                        if len(fields) < 2:
                            continue
                        sample_id = ids_by_stem.get(Path(fields[0]).name)
                        if sample_id is not None:
                            wer_values_by_id[sample_id] = float(fields[1])

            missing_wer = [safe_id for safe_id in safe_ids if safe_id not in wer_values_by_id]
            if missing_wer:
                raise RuntimeError(f"WER results are missing samples: {missing_wer}")

            mos_values_by_id = self._score_mos_resident(generated_paths, safe_ids)
            return self._result_from_metrics(
                [wer_values_by_id[safe_id] for safe_id in safe_ids],
                [sim_values_by_id[safe_id] for safe_id in safe_ids],
                [mos_values_by_id[safe_id] for safe_id in safe_ids],
                wer_weight=self.wer_weight,
                sim_weight=self.sim_weight,
                mos_weight=self.mos_weight,
            )
        finally:
            if self.keep_work_dir:
                print(f"Keeping reward work directory: {root}", flush=True)
            else:
                shutil.rmtree(root, ignore_errors=True)

    def _score_direct(
        self,
        *,
        generated_paths: list[str],
        prompt_paths: list[str],
        target_texts: list[str],
        languages: list[str | None],
        reference_paths: list[str | None],
    ) -> dict[str, torch.Tensor]:
        wer_values = []
        sim_values = []
        for generated, prompt, text, language, reference_path in zip(
            generated_paths, prompt_paths, target_texts, languages, reference_paths
        ):
            if self.wer_weight > 0:
                resolved_language = self._resolve_language(text, language)
                hypothesis = self._transcribe(generated, resolved_language)
                wer_values.append(self._wer(hypothesis, text, resolved_language))
            else:
                wer_values.append(0.0)
            if self.sim_weight > 0:
                prompt_emb = self._sim_embedding(prompt)
                generated_emb = self._sim_embedding(generated)
                sim_values.append(float(F.cosine_similarity(prompt_emb, generated_emb).item()))
            else:
                sim_values.append(0.0)
        sample_ids = [f"sample_{i:06d}" for i in range(len(generated_paths))]
        mos_values = self._score_mos_resident(generated_paths, sample_ids)
        return self._result_from_metrics(
            wer_values,
            sim_values,
            [mos_values[sample_id] for sample_id in sample_ids],
            wer_weight=self.wer_weight,
            sim_weight=self.sim_weight,
            mos_weight=self.mos_weight,
        )

    def _get_resident_worker(self, mode: str, language: str | None = None) -> _ResidentWorker:
        key = (mode, language)
        worker = self._resident_workers.get(key)
        if worker is None:
            worker = _ResidentWorker(
                mode=mode,
                language=language,
                python_executable=(
                    self.sim_python if mode == "sim"
                    else self.mos_python if mode == "mos"
                    else self.wer_python
                ),
                worker_script=self.resident_worker,
                sim_root=self.sim_root,
                device=(
                    self.sim_device if mode == "sim"
                    else self.mos_device if mode == "mos"
                    else self.wer_device
                ),
            )
            self._resident_workers[key] = worker
        return worker

    def _score_resident(
        self,
        *,
        generated_paths: list[str],
        prompt_paths: list[str],
        target_texts: list[str],
        languages: list[str | None],
        reference_paths: list[str | None],
    ) -> dict[str, torch.Tensor]:
        resolved_languages = [
            self._resolve_language(text, language)
            for text, language in zip(target_texts, languages)
        ]
        sample_ids = [
            f"{self._safe_id(Path(path).stem, index)}_{index:06d}"
            for index, path in enumerate(generated_paths)
        ]
        sim_items = [
            {
                "sample_id": sample_id,
                "generated_path": str(Path(generated).resolve()),
                "prompt_path": str(Path(prompt).resolve()),
            }
            for sample_id, generated, prompt in zip(sample_ids, generated_paths, prompt_paths)
        ]
        sim_values = (
            self._get_resident_worker("sim").request(sim_items)
            if self.sim_weight > 0
            else {sample_id: 0.0 for sample_id in sample_ids}
        )

        wer_values: dict[str, dict] = {
            sample_id: {"wer": 0.0} for sample_id in sample_ids
        }
        mos_values = self._score_mos_resident(generated_paths, sample_ids)
        for language in sorted(set(resolved_languages)) if self.wer_weight > 0 else []:
            items = [
                {
                    "sample_id": sample_ids[index],
                    "generated_path": str(Path(generated_paths[index]).resolve()),
                    "target_text": target_texts[index],
                    "reference_path": (
                        str(Path(reference_paths[index]).resolve())
                        if reference_paths[index]
                        else None
                    ),
                }
                for index, item_language in enumerate(resolved_languages)
                if item_language == language
            ]
            wer_values.update(self._get_resident_worker("wer", language).request(items))

        missing_sim = [sample_id for sample_id in sample_ids if sample_id not in sim_values]
        missing_wer = [sample_id for sample_id in sample_ids if sample_id not in wer_values]
        missing_mos = [sample_id for sample_id in sample_ids if sample_id not in mos_values]
        if missing_sim or missing_wer or missing_mos:
            raise RuntimeError(
                f"Resident reward results are incomplete: missing_sim={missing_sim}, "
                f"missing_wer={missing_wer}, missing_mos={missing_mos}"
            )
        return self._result_from_metrics(
            [float(wer_values[sample_id]["wer"]) for sample_id in sample_ids],
            [float(sim_values[sample_id]) for sample_id in sample_ids],
            [float(mos_values[sample_id]) for sample_id in sample_ids],
            wer_weight=self.wer_weight,
            sim_weight=self.sim_weight,
            mos_weight=self.mos_weight,
        )

    def close(self):
        for worker in list(self._resident_workers.values()):
            worker.close()
        self._resident_workers.clear()

    def score(
        self,
        *,
        generated_paths: list[str],
        prompt_paths: list[str],
        target_texts: list[str],
        languages: list[str | None] | None = None,
        reference_paths: list[str | None] | None = None,
    ) -> dict[str, torch.Tensor]:
        if not (
            len(generated_paths)
            == len(prompt_paths)
            == len(target_texts)
        ):
            raise ValueError("Generated paths, prompt paths and target texts must have equal length.")
        languages = languages or [None] * len(generated_paths)
        reference_paths = reference_paths or [None] * len(generated_paths)
        if len(reference_paths) != len(generated_paths):
            raise ValueError("Reference paths must have the same length as generated paths.")

        if self.backend == "resident":
            return self._score_resident(
                generated_paths=generated_paths,
                prompt_paths=prompt_paths,
                target_texts=target_texts,
                languages=languages,
                reference_paths=reference_paths,
            )
        if self.backend == "server":
            result = self._server_client.request(
                generated_paths=generated_paths,
                prompt_paths=prompt_paths,
                target_texts=target_texts,
                languages=languages,
                reference_paths=reference_paths,
            )
            return {
                key: torch.tensor(value, dtype=torch.float32)
                for key, value in result.items()
            }
        if self.backend == "script":
            return self._score_with_scripts(
                generated_paths=generated_paths,
                prompt_paths=prompt_paths,
                target_texts=target_texts,
                languages=languages,
                reference_paths=reference_paths,
            )
        if self.backend == "direct":
            return self._score_direct(
                generated_paths=generated_paths,
                prompt_paths=prompt_paths,
                target_texts=target_texts,
                languages=languages,
                reference_paths=reference_paths,
            )
        raise ValueError(f"Unknown reward backend: {self.backend}")
