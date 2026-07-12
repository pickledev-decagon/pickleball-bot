"""
Env throughput benchmark (NOT TRAINING — no model.learn, no PPO).

Steps the Unity env with a fixed neutral action [0,0,0] for N steps and
measures wall-clock time. Extrapolates to common training budgets so you
can decide whether a training run is worth committing to.

Single-env measurement. With NUM_ENVS=8 parallel envs, effective throughput
is roughly 8x this (minus subprocess overhead), so the extrapolated times
are conservative upper bounds for the parallel case.

Run:  /Users/keven/miniforge3/envs/dpickleball/bin/python scripts/benchmark_env_throughput.py
      /Users/keven/miniforge3/envs/dpickleball/bin/python scripts/benchmark_env_throughput.py --steps 2000 --port 5025
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BUILD_PATH, check_build_path, NUM_ENVS
from agent.wrapper_vec import make_unity_env_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--port", type=int, default=5025)
    args = ap.parse_args()

    if not check_build_path():
        print("\nSet DPICKLEBALL_BUILD_PATH or create .env. Aborting.")
        return

    print(f"\nBenchmark config: {args.steps} steps, single env, port {args.port}")
    print(f"Training config uses NUM_ENVS={NUM_ENVS} parallel envs.\n")

    # --- Env startup timing ---
    t0 = time.time()
    print("Starting Unity env ...")
    env_fn = make_unity_env_fn(
        rank=0, build_path=BUILD_PATH, frame_stack=4,
        use_ball_tracking=False, use_heatmap=False,
        base_port=args.port, reward_config=None, opponent_policy=None,
    )
    wrapper = env_fn()
    parallel = wrapper.env
    agents = list(parallel.possible_agents)
    t_startup = time.time() - t0
    print(f"Env startup: {t_startup:.1f}s  (agents={agents})\n")

    # --- Warmup (first few steps are slow — JIT, caching) ---
    neutral = np.array([0, 0, 0], dtype=np.int32)
    print("Warmup (20 steps) ...")
    parallel.reset()
    for _ in range(20):
        actions = {aid: neutral.copy() for aid in agents}
        parallel.step(actions)

    # --- Timed run ---
    print(f"Timed run ({args.steps} steps) ...")
    parallel.reset()
    t0 = time.time()
    n_done = 0
    for t in range(args.steps):
        actions = {aid: neutral.copy() for aid in agents}
        obs_dict, rewards, terms, infos = parallel.step(actions)
        n_done += 1
        if any(terms.values()):
            parallel.reset()
    t_run = time.time() - t0

    try:
        parallel.close()
    except Exception:
        pass

    # --- Results ---
    sps_single = n_done / t_run  # steps per second, single env
    print("\n" + "=" * 56)
    print("ENV THROUGHPUT BENCHMARK")
    print("=" * 56)
    print(f"  Steps completed:    {n_done}")
    print(f"  Wall-clock time:    {t_run:.1f}s")
    print(f"  Single-env throughput: {sps_single:.1f} steps/sec")
    print(f"  Env startup time:   {t_startup:.1f}s")
    print()

    # Extrapolation
    budgets = [
        ("Phase 1 sanity (50k)",        50_000),
        ("Current config (200k)",       200_000),
        ("Phase 2/3 ablation (100k)",   100_000),
        ("Phase 5 lower (2M)",          2_000_000),
        ("Phase 5 upper (5M)",          5_000_000),
    ]

    print("  Extrapolated wall-clock times (single env):")
    for label, steps in budgets:
        secs = steps / sps_single
        hrs = secs / 3600
        if hrs >= 1:
            print(f"    {label:32s}  {hrs:6.1f} hours")
        else:
            print(f"    {label:32s}  {secs:6.0f} sec  ({secs/60:.1f} min)")

    print()
    print(f"  With NUM_ENVS={NUM_ENVS} parallel (rough 8x estimate, conservative):")
    for label, steps in budgets:
        secs = steps / (sps_single * NUM_ENVS * 0.85)  # 0.85 = subprocess overhead factor
        hrs = secs / 3600
        if hrs >= 1:
            print(f"    {label:32s}  {hrs:6.1f} hours")
        else:
            print(f"    {label:32s}  {secs:6.0f} sec  ({secs/60:.1f} min)")

    print()
    print("  NOTE: Parallel estimate assumes 85% scaling efficiency.")
    print("        On Mac with 8 Unity instances, actual may be lower due to")
    print("        CPU/GPU contention. Real training also adds PPO overhead")
    print("        (rollout collection + gradient updates) on top of env stepping.")
    print("=" * 56)


if __name__ == "__main__":
    main()
