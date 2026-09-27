#!/usr/bin/env python3
"""Hold GPUs with dynamic utilization and configurable VRAM / process name."""

from __future__ import annotations

import argparse
import ctypes
import math
import multiprocessing as mp
import os
import random
import signal
import sys
import time
from dataclasses import dataclass


def set_process_title(title: str) -> None:
    try:
        import setproctitle

        setproctitle.setproctitle(title)
        return
    except ImportError:
        pass

    try:
        libc = ctypes.CDLL("libc.so.6")
        name = title.encode("utf-8", errors="replace")[:15]
        libc.prctl(15, name, 0, 0, 0)
    except Exception:
        pass


def parse_gpus(gpu_arg: str | None, num_gpus: int | None) -> list[int]:
    if gpu_arg:
        gpus = [int(x.strip()) for x in gpu_arg.split(",") if x.strip()]
        if not gpus:
            raise ValueError("Empty --gpus list.")
        return gpus
    if num_gpus is None:
        raise ValueError("Provide --gpus or --num-gpus.")
    return list(range(num_gpus))


def bytes_from_gib(gib: float) -> int:
    return int(gib * (1024**3))


@dataclass
class WorkerConfig:
    gpu_id: int
    vram_gib: float
    process_name: str
    min_util: float
    max_util: float
    cycle_sec: float
    seed: int


def _target_util(step: float, min_util: float, max_util: float) -> float:
    # Smooth oscillation between min/max utilization.
    mid = 0.5 * (min_util + max_util)
    amp = 0.5 * (max_util - min_util)
    return mid + amp * math.sin(step)


def _allocate_vram(device: str, gib: float):
    import torch

    if gib <= 0:
        return None

    total = bytes_from_gib(gib)
    chunk = bytes_from_gib(0.25)
    tensors = []
    allocated = 0
    while allocated + chunk <= total:
        try:
            tensors.append(torch.empty(chunk // 4, dtype=torch.float32, device=device))
            allocated += chunk
        except torch.cuda.OutOfMemoryError:
            break
    return tensors


def _run_compute(device: str, duration_sec: float, matrix_size: int):
    import torch

    if duration_sec <= 0:
        return

    a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
    b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
    deadline = time.time() + duration_sec
    while time.time() < deadline:
        c = a @ b
        c = torch.nn.functional.relu(c)
        del c


def gpu_worker(cfg: WorkerConfig, stop_event: mp.Event) -> None:
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    device = "cuda:0"
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA not available for GPU {cfg.gpu_id}")

    title = cfg.process_name if cfg.process_name else f"gpu_occupy:{cfg.gpu_id}"
    set_process_title(title)

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.set_device(0)

    vram_holder = _allocate_vram(device, cfg.vram_gib)
    actual_gib = sum(t.numel() * t.element_size() for t in (vram_holder or [])) / (1024**3)

    step = random.uniform(0, math.pi * 2)
    matrix_size = 4096

    print(
        f"[gpu {cfg.gpu_id}] started as '{title}', "
        f"target_vram={cfg.vram_gib:.1f}GiB, actual_vram~={actual_gib:.2f}GiB, "
        f"util_range=[{cfg.min_util:.0%}, {cfg.max_util:.0%}]",
        flush=True,
    )

    while not stop_event.is_set():
        target = _target_util(step, cfg.min_util, cfg.max_util)
        step += (2 * math.pi / max(cfg.cycle_sec, 1.0)) * 0.5

        compute_sec = cfg.cycle_sec * target
        sleep_sec = cfg.cycle_sec - compute_sec

        _run_compute(device, compute_sec, matrix_size)
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    del vram_holder
    torch.cuda.empty_cache()
    print(f"[gpu {cfg.gpu_id}] stopped.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Occupy GPUs with dynamic utilization.")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7", help="Comma-separated GPU ids.")
    parser.add_argument("--num-gpus", type=int, default=None, help="Use GPUs 0..N-1 if --gpus is omitted.")
    parser.add_argument("--vram-gib", type=float, default=20.0, help="VRAM to hold per GPU (GiB).")
    parser.add_argument("--process-name", type=str, default="cosyvoice2", help="Process name shown in ps/nvidia-smi.")
    parser.add_argument("--min-util", type=float, default=0.15, help="Minimum target GPU utilization.")
    parser.add_argument("--max-util", type=float, default=0.85, help="Maximum target GPU utilization.")
    parser.add_argument("--cycle-sec", type=float, default=8.0, help="One utilization oscillation period (seconds).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    if not 0.0 <= args.min_util <= 1.0 or not 0.0 <= args.max_util <= 1.0:
        parser.error("--min-util/--max-util must be in [0, 1].")
    if args.min_util > args.max_util:
        parser.error("--min-util must be <= --max-util.")
    if args.vram_gib < 0:
        parser.error("--vram-gib must be >= 0.")

    gpus = parse_gpus(args.gpus, args.num_gpus)
    stop_event = mp.Event()
    processes: list[mp.Process] = []

    def _handle_signal(signum, _frame):
        print(f"\nReceived signal {signum}, stopping workers...", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print(
        f"Launching {len(gpus)} workers on GPUs {gpus}, "
        f"process_name='{args.process_name}', vram={args.vram_gib}GiB/gpu, "
        f"util=[{args.min_util:.0%}, {args.max_util:.0%}], cycle={args.cycle_sec}s",
        flush=True,
    )

    ctx = mp.get_context("spawn")
    for idx, gpu_id in enumerate(gpus):
        cfg = WorkerConfig(
            gpu_id=gpu_id,
            vram_gib=args.vram_gib,
            process_name=args.process_name,
            min_util=args.min_util,
            max_util=args.max_util,
            cycle_sec=args.cycle_sec,
            seed=args.seed + idx,
        )
        proc = ctx.Process(target=gpu_worker, args=(cfg, stop_event), daemon=False)
        proc.start()
        processes.append(proc)

    try:
        while any(p.is_alive() for p in processes):
            for p in processes:
                p.join(timeout=1.0)
            if stop_event.is_set():
                break
    except KeyboardInterrupt:
        stop_event.set()

    stop_event.set()
    for p in processes:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)

    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
