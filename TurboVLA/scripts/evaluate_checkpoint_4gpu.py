"""Evaluate one saved LIBERO checkpoint on the four training GPUs, without training."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from turbovla.training.trainer import (
    aggregate_distributed_libero_evaluation,
    run_libero_evaluation,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_available_ram_gb", type=float, default=32.0)
    parsed = parser.parse_args()

    checkpoint_path = str(Path(parsed.checkpoint).resolve())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    global_step = int(checkpoint["global_step"])
    args = SimpleNamespace(**checkpoint["args"])
    del checkpoint
    args.eval_output_dir = str(Path(parsed.output_dir).resolve())
    args.eval_distributed = True
    args.eval_min_available_ram_gb = parsed.min_available_ram_gb
    world_size = len(args.eval_cuda_visible_devices.split(","))
    if world_size != 4:
        raise ValueError(f"Expected four training GPUs, got {args.eval_cuda_visible_devices!r}")

    print(
        f"Evaluating step {global_step}, suites={args.eval_suites}, "
        f"rollouts/task={args.eval_num_trials_per_task}, GPUs={args.eval_cuda_visible_devices}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=world_size) as pool:
        futures = {
            pool.submit(run_libero_evaluation, args, checkpoint_path, global_step, rank, world_size): rank
            for rank in range(world_size)
        }
        for future in as_completed(futures):
            rank = futures[future]
            future.result()
            print(f"rank {rank} finished all suites", flush=True)

    metrics = aggregate_distributed_libero_evaluation(args, global_step, world_size)
    result_path = Path(args.eval_output_dir) / f"step_{global_step}" / "metrics.json"
    result_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved complete metrics to {result_path}", flush=True)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
