from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from wavtts.model.utils import lens_to_mask


@dataclass
class FlowGRPORollout:
    states: list[torch.Tensor]
    actions: list[torch.Tensor]
    transition_means: list[torch.Tensor]
    times: torch.Tensor
    dts: torch.Tensor
    stds: torch.Tensor
    noise_scale: float
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


def _sde_parameters(
    times: torch.Tensor,
    noise_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the translated Flow-GRPO SDE coefficients in float32.

    The paper/official implementation integrates the reverse sigma schedule
    from high noise to low noise. WavTTS integrates t from 0 to 1, so the
    reverse sigma is sigma = 1 - t and dt_sigma is negative.
    """
    times = times.float()
    sigma = 1.0 - times[:-1]
    sigma_next = 1.0 - times[1:]
    dt_sigma = sigma_next - sigma
    dts = times[1:] - times[:-1]
    if noise_scale < 0:
        raise ValueError(f"noise_scale must be >= 0, got {noise_scale}")

    if noise_scale == 0:
        sigma_t = torch.zeros_like(sigma)
        stds = torch.zeros_like(dts)
        return sigma, dt_sigma, sigma_t, stds

    noise = torch.as_tensor(float(noise_scale), device=times.device, dtype=torch.float32)
    sigma_max = sigma_next[0]
    sigma_denom = torch.where(
        sigma == 1.0,
        1.0 - sigma_max,
        (1.0 - sigma).clamp_min(torch.finfo(torch.float32).eps),
    )
    sigma_t = noise * torch.sqrt(sigma / sigma_denom)
    stds = sigma_t * torch.sqrt((-dt_sigma).clamp_min(0.0))
    return sigma, dt_sigma, sigma_t, stds


def _sde_transition_mean(
    state: torch.Tensor,
    velocity: torch.Tensor,
    *,
    step_index: int,
    dts: torch.Tensor,
    sigma: torch.Tensor,
    dt_sigma: torch.Tensor,
    sigma_t: torch.Tensor,
    noise_scale: float,
) -> torch.Tensor:
    """Compute the Gaussian transition mean used by rollout and replay."""
    state = state.float()
    velocity = velocity.float()
    if noise_scale == 0:
        return state + dts[step_index].float() * velocity

    sigma_value = sigma[step_index].float()
    sigma_noise = sigma_t[step_index].float()
    dt_sigma_value = dt_sigma[step_index].float()
    inv_sigma = sigma_value.clamp_min(torch.finfo(torch.float32).eps)
    return (
        state
        * (1.0 + sigma_noise.square() / (2.0 * inv_sigma) * dt_sigma_value)
        + (-velocity)
        * (
            1.0
            + sigma_noise.square() * (1.0 - sigma_value)
            / (2.0 * inv_sigma)
        )
        * dt_sigma_value
    )


def _gaussian_log_prob(
    action: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor | float,
    mask: torch.Tensor,
) -> torch.Tensor:
    action = action.float()
    mean = mean.float()
    std_tensor = torch.as_tensor(std, device=action.device, dtype=torch.float32)
    if bool(torch.any(std_tensor <= 0)):
        raise ValueError(
            "Gaussian policy log-probability requires noise_scale > 0. "
            "Use CFM.sample() for deterministic ODE quality checks."
        )
    values = -0.5 * (
        ((action - mean) / std_tensor).square()
        + 2.0 * torch.log(std_tensor)
        + torch.log(torch.as_tensor(2.0 * torch.pi, device=action.device, dtype=torch.float32))
    )
    return _masked_mean(values, mask)


def rollout_log_probs(
    rollout: FlowGRPORollout,
) -> torch.Tensor:
    """Compute exact old-policy log-probs from the sampled transition means."""
    return torch.stack(
        [
            _gaussian_log_prob(action, mean, std, rollout.action_mask)
            for action, mean, std in zip(
                rollout.actions,
                rollout.transition_means,
                rollout.stds,
            )
        ],
        dim=0,
    )


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
    # The paper writes the SDE in the reverse convention (noise fraction
    # sigma=1 at the start and sigma=0 at the end). This model integrates
    # t=0 -> 1, so sigma is 1-t and dt_sigma=-dt.
    sigma, dt_sigma, sigma_t, stds = _sde_parameters(times, noise_scale)

    # Keep the stochastic state and transition arithmetic in fp32. The model
    # itself may still run under bf16 autocast.
    x = torch.randn(
        step_cond.shape,
        device=step_cond.device,
        dtype=torch.float32,
    )
    x = torch.where(valid_mask, x, torch.zeros_like(x))
    states = []
    actions = []
    transition_means = []
    for step_idx, (time_value, std) in enumerate(zip(times[:-1], stds)):
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
        mean = _sde_transition_mean(
            x,
            velocity,
            step_index=step_idx,
            dts=dts,
            sigma=sigma,
            dt_sigma=dt_sigma,
            sigma_t=sigma_t,
            noise_scale=noise_scale,
        )
        transition_means.append(mean.detach())
        action = mean if float(std.item()) == 0.0 else mean + std * torch.randn_like(mean)
        action = torch.where(valid_mask, action, torch.zeros_like(action))
        actions.append(action.detach())
        x = action

    model.transformer.clear_cache()
    final = torch.where(cond_mask, step_cond, x) / model.latents_scale
    return FlowGRPORollout(
        states=states,
        actions=actions,
        transition_means=transition_means,
        times=times[:-1],
        dts=dts,
        stds=stds,
        noise_scale=float(noise_scale),
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
    times = torch.cat([rollout.times, rollout.times[-1:] + rollout.dts[-1:]])
    sigma, dt_sigma, sigma_t, _ = _sde_parameters(times, rollout.noise_scale)
    for step_idx, (state, action, time_value, std) in enumerate(zip(
        rollout.states,
        rollout.actions,
        rollout.times,
        rollout.stds,
    )):
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
        mean = _sde_transition_mean(
            state,
            velocity,
            step_index=step_idx,
            dts=rollout.dts,
            sigma=sigma,
            dt_sigma=dt_sigma,
            sigma_t=sigma_t,
            noise_scale=rollout.noise_scale,
        )
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
    std = rollout.stds[step_index]
    times = torch.cat([rollout.times, rollout.times[-1:] + rollout.dts[-1:]])
    sigma, dt_sigma, sigma_t, _ = _sde_parameters(times, rollout.noise_scale)
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
    mean = _sde_transition_mean(
        state,
        velocity,
        step_index=step_index,
        dts=rollout.dts,
        sigma=sigma,
        dt_sigma=dt_sigma,
        sigma_t=sigma_t,
        noise_scale=rollout.noise_scale,
    )
    log_prob = _gaussian_log_prob(action, mean, std, rollout.action_mask)
    model.transformer.clear_cache()
    return log_prob, mean if return_mean else None


def slice_rollout(rollout: FlowGRPORollout, start: int, end: int) -> FlowGRPORollout:
    """Take a batch slice while keeping the shared diffusion schedule."""
    if not 0 <= start < end <= rollout.final.shape[0]:
        raise ValueError(
            f"Invalid rollout batch slice [{start}:{end}] for batch {rollout.final.shape[0]}"
        )
    return FlowGRPORollout(
        states=[value[start:end] for value in rollout.states],
        actions=[value[start:end] for value in rollout.actions],
        transition_means=[value[start:end] for value in rollout.transition_means],
        times=rollout.times,
        dts=rollout.dts,
        stds=rollout.stds,
        noise_scale=rollout.noise_scale,
        action_mask=rollout.action_mask[start:end],
        cond_mask=rollout.cond_mask[start:end],
        valid_mask=rollout.valid_mask[start:end],
        condition=rollout.condition[start:end],
        step_cond=rollout.step_cond[start:end],
        durations=rollout.durations[start:end],
        final=rollout.final[start:end],
    )


def concatenate_rollouts(rollouts: list[FlowGRPORollout]) -> FlowGRPORollout:
    """Concatenate rollout microbatches generated with one shared schedule."""
    if not rollouts:
        raise ValueError("At least one rollout is required.")
    first = rollouts[0]
    for rollout in rollouts[1:]:
        if rollout.noise_scale != first.noise_scale:
            raise ValueError("Cannot concatenate rollouts with different noise scales.")
        for name in ("times", "dts", "stds"):
            if not torch.equal(getattr(rollout, name), getattr(first, name)):
                raise ValueError(f"Cannot concatenate rollouts with different {name}.")
    return FlowGRPORollout(
        states=[torch.cat(values, dim=0) for values in zip(*(item.states for item in rollouts))],
        actions=[torch.cat(values, dim=0) for values in zip(*(item.actions for item in rollouts))],
        transition_means=[
            torch.cat(values, dim=0)
            for values in zip(*(item.transition_means for item in rollouts))
        ],
        times=first.times,
        dts=first.dts,
        stds=first.stds,
        noise_scale=first.noise_scale,
        action_mask=torch.cat([item.action_mask for item in rollouts], dim=0),
        cond_mask=torch.cat([item.cond_mask for item in rollouts], dim=0),
        valid_mask=torch.cat([item.valid_mask for item in rollouts], dim=0),
        condition=torch.cat([item.condition for item in rollouts], dim=0),
        step_cond=torch.cat([item.step_cond for item in rollouts], dim=0),
        durations=torch.cat([item.durations for item in rollouts], dim=0),
        final=torch.cat([item.final for item in rollouts], dim=0),
    )


def gaussian_mean_kl(
    mean: torch.Tensor,
    reference_mean: torch.Tensor,
    stds: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    mean = mean.float()
    reference_mean = reference_mean.float()
    stds = stds.float().reshape(-1, 1, 1).clamp_min(torch.finfo(torch.float32).eps)
    values = 0.5 * ((mean - reference_mean) / stds).square()
    mask = action_mask[None, :, :].to(dtype=values.dtype)
    return (values * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
