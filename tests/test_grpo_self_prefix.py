import json

import torch
import torchaudio

from wavtts.model.grpo_dataset import PromptTargetJsonl, sample_prefix_length
from wavtts.model.grpo_flow import _sde_parameters, _time_grid, prepare_flow_inputs
from wavtts.rewards import SimWerReward


def test_self_prefix_dataset_uses_only_prompt_fields(tmp_path):
    wav_path = tmp_path / "sample.wav"
    torchaudio.save(str(wav_path), torch.zeros(1, 16_000), 16_000)
    jsonl_path = tmp_path / "data.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "utt": "sample",
                "prompt_wav": str(wav_path),
                "prompt_token": list(range(25)),
                "target_token": [999],
                "target_text": "must be ignored",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    task = PromptTargetJsonl(str(jsonl_path), data_mode="self_prefix")[0]
    assert task["utt"] == "sample"
    assert task["duration"] == 16_000
    assert task["speech_token"].tolist() == list(range(25))
    assert task["target_text"] == ""


def test_prefix_sampling_is_stable_and_frame_aligned():
    first = sample_prefix_length(
        16_000,
        frame_len=160,
        min_fraction=0.0,
        max_fraction=0.3,
        seed=123,
    )
    second = sample_prefix_length(
        16_000,
        frame_len=160,
        min_fraction=0.0,
        max_fraction=0.3,
        seed=123,
    )
    assert first == second
    assert first[0] % 160 == 0
    assert 0 <= first[0] < 4_800


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.latents_scale = 1.0

    def _prepare_condition(self, *, text, speech_tokens, batch, device):
        return speech_tokens.to(device)

    def _prepare_spk_embedding(self, value, *, batch, device):
        return value.to(device)


def test_flow_action_mask_contains_suffix_only():
    inputs = prepare_flow_inputs(
        _DummyModel(),
        cond=torch.zeros(2, 1_600),
        speech_tokens=torch.ones(2, 3, dtype=torch.long),
        prompt_lens=torch.tensor([0, 480]),
        durations=torch.tensor([1_600, 1_600]),
        spk_embedding=None,
    )
    assert inputs["action_mask"][0].all()
    assert not inputs["action_mask"][1, :480].any()
    assert inputs["action_mask"][1, 480:].all()


def test_flow_grpo_first_step_noise_matches_reference_equation():
    times = _time_grid(10, device=torch.device("cpu"))
    _, _, _, stds = _sde_parameters(times, 0.1)
    assert torch.isclose(stds[0], torch.tensor(0.1), atol=1e-6)


def test_sim_only_reward_does_not_start_wer_worker(monkeypatch, tmp_path):
    reward = SimWerReward(
        sim_root=str(tmp_path),
        wer_weight=0.0,
        sim_weight=1.0,
        backend="resident",
    )
    requested_modes = []

    class Worker:
        def request(self, items):
            return {item["sample_id"]: 0.8 for item in items}

    def worker(mode, language=None):
        requested_modes.append(mode)
        assert mode == "sim"
        return Worker()

    monkeypatch.setattr(reward, "_get_resident_worker", worker)
    result = reward.score(
        generated_paths=[str(tmp_path / "generated.wav")],
        prompt_paths=[str(tmp_path / "prompt.wav")],
        target_texts=[""],
        languages=["zh"],
    )
    assert requested_modes == ["sim"]
    assert result["wer"].item() == 0.0
    assert torch.isclose(result["sim"], torch.tensor([0.8])).all()
