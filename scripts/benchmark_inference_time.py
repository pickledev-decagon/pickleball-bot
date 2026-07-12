"""
Inference timing benchmark — competition deadline check.

The competition requires 0.01s (10ms) response time per action.
This script loads the Phase 5 checkpoint and times predictions on CPU
to verify we meet the deadline.

Usage:
  python scripts/benchmark_inference_time.py
  python scripts/benchmark_inference_time.py --checkpoint path/to/ckpt.zip --n 1000
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO


def main():
    ap = argparse.ArgumentParser(description="Inference timing benchmark")
    ap.add_argument("--checkpoint", type=str,
                    default="checkpoints_selfplay_v2/phase5_selfplay_20260628_050245/phase5_500000_steps.zip",
                    help="Path to checkpoint")
    ap.add_argument("--n", type=int, default=1000,
                    help="Number of predictions to time")
    ap.add_argument("--device", type=str, default="cpu",
                    help="Device for inference (cpu for competition)")
    args = ap.parse_args()

    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ckpt_path)
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    print(f"\n{'='*60}")
    print(f"INFERENCE TIMING BENCHMARK — Competition Deadline Check")
    print(f"{'='*60}")
    print(f"  Checkpoint: {os.path.basename(ckpt_path)}")
    print(f"  Device:     {args.device} (competition uses CPU)")
    print(f"  Predictions: {args.n}")
    print(f"  Deadline:   10ms (0.01s) per action")
    print(f"{'='*60}\n")

    # Load model
    print("Loading model ...")
    model = PPO.load(ckpt_path, device=args.device)

    # Get observation shape from model
    obs_shape = model.observation_space.shape
    print(f"  Observation shape: {obs_shape}")

    # Create dummy observation (same shape as training obs)
    dummy_obs = np.zeros(obs_shape, dtype=np.float32)

    # Warmup — first prediction is always slow (lazy init)
    print("Warming up (10 predictions) ...")
    for _ in range(10):
        model.predict(dummy_obs, deterministic=False)

    # Timed predictions
    print(f"Timing {args.n} predictions ...")
    times = []
    for i in range(args.n):
        t0 = time.perf_counter()
        action, _ = model.predict(dummy_obs, deterministic=False)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # convert to ms

    times = np.array(times)
    mean_ms = np.mean(times)
    median_ms = np.median(times)
    p95_ms = np.percentile(times, 95)
    p99_ms = np.percentile(times, 99)
    max_ms = np.max(times)
    min_ms = np.min(times)

    print(f"\n{'='*60}")
    print(f"RESULTS — {args.n} predictions on {args.device}")
    print(f"{'='*60}")
    print(f"  Mean:    {mean_ms:.2f} ms")
    print(f"  Median:  {median_ms:.2f} ms")
    print(f"  P95:     {p95_ms:.2f} ms")
    print(f"  P99:     {p99_ms:.2f} ms")
    print(f"  Min:     {min_ms:.2f} ms")
    print(f"  Max:     {max_ms:.2f} ms")
    print(f"  Deadline: 10.00 ms")
    print(f"{'='*60}")

    # Verdict
    print(f"\nVerdict:")
    if mean_ms < 10:
        print(f"  ✓ PASS — Mean inference time {mean_ms:.2f}ms < 10ms deadline")
    else:
        print(f"  ✗ FAIL — Mean inference time {mean_ms:.2f}ms > 10ms deadline")

    if p99_ms < 10:
        print(f"  ✓ P99 ({p99_ms:.2f}ms) under deadline — rare slow cases OK")
    else:
        print(f"  ⚠ P99 ({p99_ms:.2f}ms) exceeds deadline — 1% of predictions too slow")

    if max_ms < 10:
        print(f"  ✓ Max ({max_ms:.2f}ms) under deadline — worst case OK")
    else:
        print(f"  ⚠ Max ({max_ms:.2f}ms) exceeds deadline — worst case too slow (may be GC/scheduling)")

    # Competition realism note
    print(f"\nNote: Competition runs on unknown hardware. These times are on your Mac.")
    print(f"  If mean < 5ms, likely safe. If 5-10ms, risky on slower hardware.")
    print(f"  If > 10ms, need model distillation or smaller CNN.")

    # Also test with deterministic=True (competition may use either)
    print(f"\nTesting deterministic=True ...")
    times_det = []
    for _ in range(10):
        model.predict(dummy_obs, deterministic=True)
    for i in range(args.n):
        t0 = time.perf_counter()
        action, _ = model.predict(dummy_obs, deterministic=True)
        t1 = time.perf_counter()
        times_det.append((t1 - t0) * 1000)

    times_det = np.array(times_det)
    mean_det = np.mean(times_det)
    print(f"  Deterministic=True mean: {mean_det:.2f} ms (vs {mean_ms:.2f} ms stochastic)")


if __name__ == "__main__":
    main()
