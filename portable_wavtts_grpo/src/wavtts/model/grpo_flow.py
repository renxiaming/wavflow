from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from wavtts.model.utils import lens_to_mask


@dataclass
class FlowGRPORollout:
    states: list[torch.Tensor]
    actions: list[torch.Tensor]
    times: torch.Tensor
    dts: torch.Tensor
    stds: torch.Tensor
    action_mask: torch.Tensor
    cond_mask: torch.Tensor
    valid_mask: torch.Tensor
    condition: torch.Tensor
    step_cond: torch.Tensor
    durations: torch.Tensor
    final: torch.Tensor


def _time_grid(
    steps: int,
    *,
    device: torch.device,
    mapping: str = "uniform",
    power: float | None = None,
    sway_sampling_coef: float | None = None,
    shift: float = 1.0,
) -> torch.Tensor:
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    t = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=torch.float32)
    if mapping == "uniform":
        pass
    elif mapping == "power":
        if power is None:
            raise ValueError("timestep power is required for mapping='power'")
        t = t.pow(max(float(power), 5e-2))
    elif mapping == "sway_sampling":
        coef = -1.0 if sway_sampling_coef is None else float(sway_sampling_coef)
        t = t + coef * (torch.cos(torch.pi / 2 * t) - 1.0 + t)
    else:
        raise ValueError(f"Unknown timestep mapping: {mapping}")

    effective_shift = max(float(shift), 5e-2)
    if effective_shift != 1.0:
        denom = t + effective_shift * (1.0 - t)
        t = t / denom.clamp_min(torch.finfo(t.dtype).eps)

    if not bool(torch.all(t[1:] > t[:-1])):
        raise ValueError("The generated timestep grid is not strictly increasing.")
    return t


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype)
    return (values * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)


def _gaussian_log_prob(
    action: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor | float,
    mask: torch.Tensor,
) -> torch.Tensor:
    std_tensor = torch.as_tensor(std, device=action.device, dtype=action.dtype)
    values = -0.5 * (
        ((action - mean) / std_tensor).square()
        + 2.0 * torch.log(std_tensor)
        + torch.log(torch.as_tensor(2.0 * torch.pi, device=action.device, dtype=action.dtype))
    )
    return _masked_mean(values, mask)


def _to_velocity(model, raw_pred: torch.Tensor, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    if model.prediction == "flow":
        return raw_pred
    if model.prediction == "x_pred":
        return model._x_to_v(raw_pred, x, time)
    raise ValueError(f"Unknown model prediction mode: {model.prediction}")


def predict_velocity(
    model,
    *,
    x: torch.Tensor,
    condition: torch.Tensor,
    step_cond: torch.Tensor,
    time: torch.Tensor,
    valid_mask: torch.Tensor,
    durations: torch.Tensor,
    spk_embedding: torch.Tensor | None,
    cfg_strength: float,
) -> torch.Tensor:
    """Evaluate the fixed policy mean field used by both rollout and replay."""
    kwargs = dict(
        x=x,
        cond=step_cond,
        text=condition,
        time=time,
        mask=valid_mask,
        lens=durations,
        cache=False,
        spk_embedding=spk_embedding,
    )

    if cfg_strength < 1e-5:
        raw_pred, *_ = model._transformer_forward(
            **kwargs,
            drop_audio_cond=True,
            drop_text=True,
            drop_spk=True,
        )
        return _to_velocity(model, raw_pred, x, time)

    if abs(float(cfg_strength) - 1.0) < 1e-6:
        raw_pred, *_ = model._transformer_forward(
            **kwargs,
            drop_audio_cond=False,
            drop_text=False,
            drop_spk=False,
        )
        return _to_velocity(model, raw_pred, x, time)

    raw_cfg, *_ = model._transformer_forward(
        **kwargs,
        cfg_infer=True,
        drop_spk=False,
    )
    cond_pred, uncond_pred = torch.chunk(raw_cfg, 2, dim=0)
    cond_v = _to_velocity(model, cond_pred, x, time)
    uncond_v = _to_velocity(model, uncond_pred, x, time)
    return uncond_v + float(cfg_strength) * (cond_v - uncond_v)


def prepare_flow_inputs(
    model,
    *,
    cond: torch.Tensor,
    speech_tokens: torch.Tensor,
    prompt_lens: torch.Tensor,
    durations: torch.Tensor,
    spk_embedding: torch.Tensor | None,
) -> dict[str, torch.Tensor | None]:
    if cond.ndim != 2:
        raise ValueError(f"cond must be [B, N], got {tuple(cond.shape)}")
    if speech_tokens.ndim != 2:
        raise ValueError(f"speech_tokens must be [B, T], got {tuple(speech_tokens.shape)}")

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    batch = cond.shape[0]
    durations = durations.to(device=device, dtype=torch.long)
    prompt_lens = prompt_lens.to(device=device, dtype=torch.long)
    if durations.shape != (batch,) or prompt_lens.shape != (batch,):
        raise ValueError("durations and prompt_lens must be [B].")
    if torch.any(durations <= prompt_lens):
        raise ValueError("Every duration must be larger than its prompt length.")

    condition = model._prepare_condition(
        text=None,
        speech_tokens=speech_tokens,
        batch=batch,
        device=device,
    )
    cond = cond.to(device=device, dtype=dtype) * model.latents_scale
    max_duration = int(durations.max().item())
    if cond.shape[1] > max_duration:
        raise ValueError("Prompt waveform is longer than the requested duration.")
    cond = F.pad(cond, (0, max_duration - cond.shape[1]), value=0.0)
    cond_mask = lens_to_mask(prompt_lens, length=max_duration)
    valid_mask = lens_to_mask(durations, length=max_duration)
    step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))
    action_mask = valid_mask & ~cond_mask

    if spk_embedding is not None:
        spk_embedding = model._prepare_spk_embedding(spk_embedding, batch=batch, device=device)

    return {
        "condition": condition,
        "cond": cond,
        "step_cond": step_cond,
        "cond_mask": cond_mask,
        "valid_mask": valid_mask,
        "action_mask": action_mask,
        "durations": durations,
        "spk_embedding": spk_embedding,
    }


@torch.no_grad()
def rollout_sde(
    model,
    *,
    cond: torch.Tensor,
    speech_tokens: torch.Tensor,
    prompt_lens: torch.Tensor,
    durations: torch.Tensor,
    spk_embedding: torch.Tensor | None,
    steps: int,
    noise_scale: float,
    cfg_strength: float = 1.0,
    timestep_mapping: str = "uniform",
    timestep_power: float | None = None,
    sway_sampling_coef: float | None = None,
    shift: float = 1.0,
) -> FlowGRPORollout:
    inputs = prepare_flow_inputs(
        model,
        cond=cond,
        speech_tokens=speech_tokens,
        prompt_lens=prompt_lens,
        durations=durations,
        spk_embedding=spk_embedding,
    )
    condition = inputs["condition"]
    step_cond = inputs["step_cond"]
    cond_mask = inputs["cond_mask"]
    valid_mask = inputs["valid_mask"]
    action_mask = inputs["action_mask"]
    durations = inputs["durations"]
    spk_embedding = inputs["spk_embedding"]

    times = _time_grid(
        steps,
        device=step_cond.device,
        mapping=timestep_mapping,
        power=timestep_power,
        sway_sampling_coef=sway_sampling_coef,
        shift=shift,
    )
    dts = times[1:] - times[:-1]
    stds = torch.as_tensor(noise_scale, device=step_cond.device, dtype=step_cond.dtype) * dts.sqrt()
    stds = stds.clamp_min(torch.finfo(step_cond.dtype).eps)

    x = torch.randn_like(step_cond)
    x = torch.where(valid_mask, x, torch.zeros_like(x))
    states = []
    actions = []
    for step_idx, (time_value, dt, std) in enumerate(zip(times[:-1], dts, stds)):
        states.append(x.detach())
        time = torch.full(
            (x.shape[0],),
            float(time_value.item()),
            device=x.device,
            dtype=torch.float32,
        )
        velocity = predict_velocity(
            model,
            x=x,
            condition=condition,
            step_cond=step_cond,
            time=time,
            valid_mask=valid_mask,
            durations=durations,
            spk_embedding=spk_embedding,
            cfg_strength=cfg_strength,
        )
        mean = x + dt.to(dtype=x.dtype) * velocity
        action = mean + std * torch.randn_like(mean)
        action = torch.where(valid_mask, action, torch.zeros_like(action))
        actions.append(action.detach())
        x = action

    model.transformer.clear_cache()
    final = torch.where(cond_mask, step_cond, x) / model.latents_scale
    return FlowGRPORollout(
        states=states,
        actions=actions,
        times=times[:-1],
        dts=dts,
        stds=stds,
        action_mask=action_mask,
        cond_mask=cond_mask,
        valid_mask=valid_mask,
        condition=condition,
        step_cond=step_cond,
        durations=durations,
        final=final,
    )


def replay_log_probs(
    model,
    rollout: FlowGRPORollout,
    *,
    spk_embedding: torch.Tensor | None,
    cfg_strength: float,
    return_means: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    log_probs = []
    means = []
    for state, action, time_value, dt, std in zip(
        rollout.states,
        rollout.actions,
        rollout.times,
        rollout.dts,
        rollout.stds,
    ):
        time = torch.full(
            (state.shape[0],),
            float(time_value.item()),
            device=state.device,
            dtype=torch.float32,
        )
        velocity = predict_velocity(
            model,
            x=state,
            condition=rollout.condition,
            step_cond=rollout.step_cond,
            time=time,
            valid_mask=rollout.valid_mask,
            durations=rollout.durations,
            spk_embedding=spk_embedding,
            cfg_strength=cfg_strength,
        )
        mean = state + dt.to(dtype=state.dtype) * velocity
        log_probs.append(_gaussian_log_prob(action, mean, std, rollout.action_mask))
        if return_means:
            means.append(mean)

    model.transformer.clear_cache()
    stacked = torch.stack(log_probs, dim=0)
    mean_stack = torch.stack(means, dim=0) if return_means else None
    return stacked, mean_stack


def replay_log_prob_step(
    model,
    rollout: FlowGRPORollout,
    step_index: int,
    *,
    spk_embedding: torch.Tensor | None,
    cfg_strength: float,
    return_mean: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    state = rollout.states[step_index]
    action = rollout.actions[step_index]
    time_value = rollout.times[step_index]
    dt = rollout.dts[step_index]
    std = rollout.stds[step_index]
    time = torch.full(
        (state.shape[0],),
        float(time_value.item()),
        device=state.device,
        dtype=torch.float32,
    )
    velocity = predict_velocity(
        model,
        x=state,
        condition=rollout.condition,
        step_cond=rollout.step_cond,
        time=time,
        valid_mask=rollout.valid_mask,
        durations=rollout.durations,
        spk_embedding=spk_embedding,
        cfg_strength=cfg_strength,
    )
    mean = state + dt.to(dtype=state.dtype) * velocity
    log_prob = _gaussian_log_prob(action, mean, std, rollout.action_mask)
    model.transformer.clear_cache()
    return log_prob, mean if return_mean else None


def gaussian_mean_kl(
    mean: torch.Tensor,
    reference_mean: torch.Tensor,
    stds: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    values = 0.5 * ((mean - reference_mean) / stds[:, None, None]).square()
    mask = action_mask[None, :, :].to(dtype=values.dtype)
    return (values * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
