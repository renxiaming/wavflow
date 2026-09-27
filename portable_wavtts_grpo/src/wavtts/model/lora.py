from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.enabled = True
        self.dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Linear(
            base.in_features,
            self.rank,
            bias=False,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        self.lora_B = nn.Linear(
            self.rank,
            base.out_features,
            bias=False,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        nn.init.normal_(self.lora_A.weight, std=1.0 / self.rank)
        nn.init.zeros_(self.lora_B.weight)
        self.base.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base(x)
        if not self.enabled:
            return output
        return output + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


@dataclass(frozen=True)
class LoRAStats:
    modules: int
    trainable_parameters: int
    total_parameters: int


def _resolve_parent(root: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def inject_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    target_suffixes: tuple[str, ...] = (
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
    ),
) -> LoRAStats:
    model.requires_grad_(False)
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and any(name.endswith(suffix) for suffix in target_suffixes)
    ]
    if not targets:
        raise RuntimeError(f"No Linear modules matched LoRA targets {target_suffixes}.")

    for name, module in targets:
        parent, child_name = _resolve_parent(model, name)
        wrapped = LoRALinear(
            module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        if child_name.isdigit():
            parent[int(child_name)] = wrapped
        else:
            setattr(parent, child_name, wrapped)

    return LoRAStats(
        modules=len(targets),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_parameters=sum(p.numel() for p in model.parameters()),
    )


def lora_modules(model: nn.Module):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module


@contextmanager
def lora_disabled(model: nn.Module):
    modules = list(lora_modules(model))
    previous = [module.enabled for module in modules]
    try:
        for module in modules:
            module.enabled = False
        yield
    finally:
        for module, enabled in zip(modules, previous):
            module.enabled = enabled


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor
        for name, tensor in model.state_dict().items()
        if ".lora_A." in name or ".lora_B." in name
    }
