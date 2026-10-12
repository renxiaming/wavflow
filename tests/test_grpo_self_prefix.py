import json
from pathlib import Path

import torch
import torchaudio
from omegaconf import OmegaConf

from wavtts.model.grpo_dataset import PromptTargetJsonl, sample_prefix_length
from wavtts.model.grpo_flow import _sde_parameters, _time_grid, prepare_flow_inputs
from wavtts.rewards import SimWerReward
from wavtts.train.train_grpo import normalize_and_fuse_rewards, save_rollout_audio


def test_flow_grpo_default_uses_paired_prompt_target():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "src/wavtts/configs/WavTTS_xpred_spkconcat_flow_grpo.yaml"
    )
    assert OmegaConf.load(config_path).grpo.data_mode == "paired"


def test_paired_dataset_keeps_full_prompt_and_target_fields(tmp_path):
    wav_path = tmp_path / "prompt.wav"
    audio = torch.arange(32_080, dtype=torch.float32).unsqueeze(0) / 32_080
    torchaudio.save(str(wav_path), audio, 16_000)
    jsonl_path = tmp_path / "data.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "utt": "target",
                "prompt_wav": str(wav_path),
                "prompt_token": [1, 2],
                "target_token": [3, 4, 5],
                "prompt_text": "prompt words",
                "target_text": "target words",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    task = PromptTargetJsonl(str(jsonl_path), data_mode="paired")[0]
    assert task["prompt_duration"] == 32_000
    assert task["prompt_wav"].numel() == task["prompt_duration"]
    assert task["prompt_token"].tolist() == [1, 2]
    assert task["target_token"].tolist() == [3, 4, 5]
    assert task["target_text"] == "target words"
    assert task["target_duration"] == 1_920

    inputs = prepare_flow_inputs(
        _DummyModel(),
        cond=task["prompt_wav"].unsqueeze(0),
        speech_tokens=torch.cat([task["prompt_token"], task["target_token"]]).unsqueeze(0),
        prompt_lens=torch.tensor([task["prompt_duration"]]),
        durations=torch.tensor([task["prompt_duration"] + task["target_duration"]]),
        spk_embedding=None,
    )
    assert not inputs["action_mask"][0, : task["prompt_duration"]].any()
    assert inputs["action_mask"][0, task["prompt_duration"] :].all()


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


def test_self_prefix_reward_audio_reconstructs_original_prefix(tmp_path):
    task = {
        "utt": "sample",
        "prompt_duration": 4,
        "audio": torch.tensor([0.1, 0.2, 0.3, 0.4, 9.0, 9.0]),
    }
    rollout = type(
        "Rollout",
        (),
        {
            "durations": torch.tensor([6]),
            "final": torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.25, 0.3]]),
        },
    )()

    [path] = save_rollout_audio(
        rollout,
        task=task,
        output_dir=tmp_path,
        update=1,
        sample_rate=16_000,
        keep=True,
        output_gain=2.0,
        reconstruct_full=True,
    )
    saved, sample_rate = torchaudio.load(path)
    assert sample_rate == 16_000
    assert torch.allclose(saved[0], torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))


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


def test_multi_objective_reward_is_normalized_before_fusion():
    components = torch.tensor(
        [[0.9, 0.7, 3.0], [0.8, 0.8, 4.0], [0.7, 0.9, 5.0]]
    )
    fused, means, stds = normalize_and_fuse_rewards(
        components,
        torch.tensor([1.0, 1.0, 0.4]),
    )
    assert torch.isfinite(fused).all()
    assert torch.isclose(fused.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(means, torch.tensor([0.8, 0.8, 4.0]))
    assert (stds > 0).all()


def test_constant_reward_component_is_ignored():
    components = torch.tensor([[1.0, 0.2, 3.0], [1.0, 0.8, 3.0]])
    fused, _, _ = normalize_and_fuse_rewards(
        components,
        torch.tensor([1.0, 1.0, 0.4]),
    )
    assert torch.allclose(fused, torch.tensor([-1.0, 1.0]))


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
