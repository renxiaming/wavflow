"""Training-only representation alignment utilities.

The teacher is intentionally kept outside CFM.  It is frozen, is not part of
the optimizer or EMA, and is removed from the inference path automatically.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RepaAlignmentHead(nn.Module):
    """Project DiT token features into the frozen WavLM feature space."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512):
        super().__init__()
        groups = min(8, hidden_dim)
        while hidden_dim % groups != 0:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.Mish(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.Mish(),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class WavLMTeacher(nn.Module):
    """Frozen WavLM feature extractor used only for REPA supervision."""

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        layer: int | None = None,
        sample_rate: int = 16000,
    ):
        super().__init__()
        if sample_rate != 16000:
            raise ValueError("WavLM REPA teacher expects 16 kHz waveforms")

        try:
            from transformers import AutoFeatureExtractor, AutoModel
        except ImportError as exc:
            raise ImportError("transformers is required when REPA is enabled") from exc

        self.model_name = model_name
        # None means the final WavLM output after the complete encoder
        # forward pass. An integer remains available for layer ablations.
        self.layer = None if layer is None else int(layer)
        self.sample_rate = int(sample_rate)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

        num_hidden_states = int(getattr(self.encoder.config, "num_hidden_layers", 0)) + 1
        if self.layer is not None and not -num_hidden_states <= self.layer < num_hidden_states:
            raise ValueError(
                f"WavLM layer {self.layer} is out of range for {model_name}; "
                f"valid range is {-num_hidden_states}..{num_hidden_states - 1}"
            )
        self.output_dim = int(self.encoder.config.hidden_size)

    def train(self, mode: bool = True):
        # The teacher must remain in eval mode even when Trainer calls train().
        super().train(False)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(
        self,
        wav: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if wav.ndim != 2:
            raise ValueError(f"WavLMTeacher expects [B, N] waveforms, got {tuple(wav.shape)}")

        if lengths is None:
            lengths = torch.full(
                (wav.shape[0],), wav.shape[1], device=wav.device, dtype=torch.long
            )
        lengths = lengths.to(device=wav.device, dtype=torch.long).clamp(min=1, max=wav.shape[1])

        # Feature extraction is CPU-side in transformers.  Passing unpadded
        # utterances avoids teaching WavLM about the trainer's right padding.
        waveforms = [
            wav[index, : int(length.item())].detach().float().cpu().numpy()
            for index, length in enumerate(lengths)
        ]
        inputs = self.feature_extractor(
            waveforms,
            sampling_rate=self.sample_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(device=wav.device, dtype=torch.float32)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=wav.device)

        outputs = self.encoder(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=self.layer is not None,
            return_dict=True,
        )
        if self.layer is None:
            # WavLM's last_hidden_state is the output after the full encoder
            # stack (and any configured adapter), rather than an intermediate
            # hidden_states entry.
            features = outputs.last_hidden_state
        else:
            features = outputs.hidden_states[self.layer]

        if attention_mask is None:
            feature_mask = torch.ones(
                features.shape[:2], device=features.device, dtype=torch.bool
            )
        else:
            feature_mask = self.encoder._get_feature_vector_attention_mask(
                features.shape[1], attention_mask
            ).bool()

        return features, feature_mask


def repa_cosine_loss(
    generator_hidden: torch.Tensor,
    teacher_features: torch.Tensor,
    teacher_mask: torch.Tensor,
    alignment_head: nn.Module,
    source_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cosine alignment after temporal interpolation, averaged on valid frames."""

    if generator_hidden.ndim != 3 or teacher_features.ndim != 3:
        raise ValueError("REPA features must have shape [B, T, D]")
    if teacher_features.shape[0] != generator_hidden.shape[0]:
        raise ValueError("REPA teacher and generator batch sizes must match")

    target_len = teacher_features.shape[1]
    hidden = F.interpolate(
        generator_hidden.transpose(1, 2),
        size=target_len,
        mode="linear",
        align_corners=False,
    )
    projected = alignment_head(hidden).transpose(1, 2).float()
    teacher_features = teacher_features.float()

    projected = F.normalize(projected, dim=-1)
    teacher_features = F.normalize(teacher_features, dim=-1)
    valid_mask = teacher_mask.to(device=projected.device, dtype=torch.bool)

    if source_mask is not None:
        source_mask = F.interpolate(
            source_mask.float().unsqueeze(1),
            size=target_len,
            mode="nearest",
        ).squeeze(1).bool()
        valid_mask &= source_mask

    cosine_distance = 1.0 - (projected * teacher_features).sum(dim=-1)
    valid_count = valid_mask.sum(dim=1).clamp_min(1)
    per_sample = (cosine_distance * valid_mask).sum(dim=1) / valid_count
    return per_sample.mean()
