"""Waveform perceptual losses used by the training-only VAPA branch."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MultiResolutionSTFTLoss(nn.Module):
    """WaveFM-style refined multi-resolution STFT distance.

    The distance combines phase, log magnitude, local spectral gradients, and
    Laplacian terms.  It returns one value per waveform so VAPA can apply the
    per-example velocity-aware factor before averaging.
    """

    def __init__(
        self,
        fft_sizes: tuple[int, ...] = (1024, 2048, 512),
        hop_sizes: tuple[int, ...] = (128, 256, 64),
        win_sizes: tuple[int, ...] = (512, 1024, 256),
        eps: float = 1e-5,
    ):
        super().__init__()
        if not (len(fft_sizes) == len(hop_sizes) == len(win_sizes)):
            raise ValueError("STFT fft, hop, and window sizes must have equal length")
        self.fft_sizes = tuple(int(value) for value in fft_sizes)
        self.hop_sizes = tuple(int(value) for value in hop_sizes)
        self.win_sizes = tuple(int(value) for value in win_sizes)
        self.eps = float(eps)
        for index, win_size in enumerate(self.win_sizes):
            self.register_buffer(
                f"window_{index}", torch.hann_window(win_size), persistent=False
            )

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return values.mean(dim=tuple(range(1, values.ndim)))
        if mask.ndim == values.ndim - 1:
            mask = mask.unsqueeze(1)
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(values)
        denominator = mask.sum(dim=tuple(range(1, values.ndim))).clamp_min(1)
        return (values * mask).sum(dim=tuple(range(1, values.ndim))) / denominator

    @staticmethod
    def _time_mask(
        frame_mask: torch.Tensor | None,
        frame_lengths: torch.Tensor | None,
        batch: int,
        frames: int,
        hop: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if frame_mask is None and frame_lengths is None:
            return None

        if frame_lengths is None:
            frame_lengths = torch.full(
                (batch,), frame_mask.shape[1], device=device, dtype=torch.long
            )
        frame_lengths = frame_lengths.to(device=device, dtype=torch.long)

        starts = torch.zeros(batch, device=device, dtype=torch.long)
        ends = frame_lengths
        if frame_mask is not None:
            frame_mask = frame_mask.to(device=device, dtype=torch.bool)
            for index in range(batch):
                valid = frame_mask[index, : int(frame_lengths[index].item())].nonzero(
                    as_tuple=False
                ).squeeze(-1)
                if valid.numel() == 0:
                    ends[index] = 0
                else:
                    starts[index] = valid[0]
                    ends[index] = valid[-1] + 1

        frame_positions = torch.arange(frames, device=device).unsqueeze(0) * hop
        start_samples = starts.unsqueeze(1)
        end_samples = ends.unsqueeze(1)
        return (frame_positions >= start_samples) & (frame_positions < end_samples)

    def _distance_at_resolution(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        fft_size: int,
        hop_size: int,
        win_size: int,
        window: torch.Tensor,
        frame_mask: torch.Tensor | None,
        frame_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        min_length = max(fft_size, win_size)
        if x_pred.shape[-1] < min_length:
            padding = min_length - x_pred.shape[-1]
            x_pred = F.pad(x_pred, (0, padding))
            x_true = F.pad(x_true, (0, padding))

        window = window.to(device=x_pred.device, dtype=x_pred.dtype)
        pred_spec = torch.stft(
            x_pred,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_size,
            window=window,
            center=True,
            return_complex=True,
        )
        true_spec = torch.stft(
            x_true,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_size,
            window=window,
            center=True,
            return_complex=True,
        )

        pred_mag = pred_spec.abs().clamp_min(self.eps)
        true_mag = true_spec.abs().clamp_min(self.eps)
        pred_log_mag = pred_mag.log()
        true_log_mag = true_mag.log()

        # 1 - cosine of phase difference is better behaved than raw angle L1.
        pred_phase = pred_spec / pred_mag
        true_phase = true_spec / true_mag
        phase = 1.0 - (pred_phase * true_phase.conj()).real
        log_mag = (pred_log_mag - true_log_mag).abs()

        time_mask = self._time_mask(
            frame_mask,
            frame_lengths,
            x_pred.shape[0],
            pred_spec.shape[-1],
            hop_size,
            x_pred.device,
        )
        base = self._masked_mean(phase, time_mask) + self._masked_mean(log_mag, time_mask)

        time_gradient = self._masked_mean(
            (pred_log_mag[..., 1:] - pred_log_mag[..., :-1]
             - true_log_mag[..., 1:] + true_log_mag[..., :-1]).abs(),
            None if time_mask is None else time_mask[..., 1:] & time_mask[..., :-1],
        )
        frequency_gradient = self._masked_mean(
            (pred_log_mag[:, 1:, :] - pred_log_mag[:, :-1, :]
             - true_log_mag[:, 1:, :] + true_log_mag[:, :-1, :]).abs(),
            time_mask,
        )

        pred_laplacian = (
            pred_log_mag[..., 2:] - 2 * pred_log_mag[..., 1:-1] + pred_log_mag[..., :-2]
        )
        true_laplacian = (
            true_log_mag[..., 2:] - 2 * true_log_mag[..., 1:-1] + true_log_mag[..., :-2]
        )
        laplacian = self._masked_mean(
            (pred_laplacian - true_laplacian).abs(),
            None if time_mask is None else time_mask[..., 2:] & time_mask[..., 1:-1] & time_mask[..., :-2],
        )
        return (base + time_gradient + frequency_gradient + laplacian) / 5.0

    def forward(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
        frame_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_pred.ndim == 3 and x_pred.shape[1] == 1:
            x_pred = x_pred[:, 0]
        if x_true.ndim == 3 and x_true.shape[1] == 1:
            x_true = x_true[:, 0]
        if x_pred.ndim != 2 or x_true.ndim != 2:
            raise ValueError("STFT loss expects waveforms shaped [B, N]")

        # STFT kernels are not reliably implemented for bf16 on every CUDA
        # version; the cast is local and gradients still reach x_pred.
        x_pred = x_pred.float()
        x_true = x_true.float()
        losses = []
        for index, (fft_size, hop_size, win_size) in enumerate(
            zip(self.fft_sizes, self.hop_sizes, self.win_sizes)
        ):
            losses.append(
                self._distance_at_resolution(
                    x_pred,
                    x_true,
                    fft_size,
                    hop_size,
                    win_size,
                    getattr(self, f"window_{index}"),
                    frame_mask,
                    frame_lengths,
                )
            )
        return torch.stack(losses, dim=0).mean(dim=0)
