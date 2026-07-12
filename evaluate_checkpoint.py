"""
Evaluate a trained checkpoint with ground-truth match performance metrics.

Measures:
  - Win rate vs no_op, random, square opponents (both sides)
  - Spin rate (fraction of steps with non-neutral rotation)
  - Rally length (avg steps per match)
  - Point differential

Usage:
    # Evaluate with defaults (frame_stack=4, no heatmap — for baseline/sparse runs)
    python evaluate_checkpoint.py results/baseline_20260626_120400.zip

    # With heatmap (must match training config)
    python evaluate_checkpoint.py results/heatmap_20260626_120400.zip \
        --frame-stack 16 --heatmap --heatmap-sigma 5 --heatmap-weight 0.7

    # Quick eval (fewer matches)
    python evaluate_checkpoint.py results/baseline.zip --matches 20

Output: results/<checkpoint_name>_eval.json
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from stable_baselines3 import PPO
from agent.eval_harness import MatchEvaluator
from agent.wrapper_vec import make_unity_env_fn
from config import BUILD_PATH, RESULTS_DIR


def compute_spin_rate(model, env_fn, num_episodes=3, max_steps_per_ep=500):
    """Quantify paddle spinning over several evaluation episodes.

    Creates a fresh env and measures what fraction of actions have
    non-neutral rotation (action[2] != 0).

    Returns
    -------
    dict
        spin_rate:  fraction of steps where rotation action != 0
        total_steps: number of steps measured
    """
    env = env_fn()
    env.apply_reward_shaping = False
    spin_count = 0
    total_steps = 0

    try:
        for ep in range(num_episodes):
            obs, _ = env.reset()
            done = False
            for _ in range(max_steps_per_ep):
                if done:
                    break
                action, _ = model.predict(obs, deterministic=True)
                if action[2] != 0:
                    spin_count += 1
                total_steps += 1
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
    finally:
        env.close()

    total = max(total_steps, 1)
    return {
        "spin_rate": round(spin_count / total, 4),
        "spin_steps": spin_count,
        "total_steps": total_steps,
    }


def run_full_evaluation(model_path, n_matches=50, device="cpu",
                        frame_stack=4, use_heatmap=False,
                        heatmap_sigma=5, heatmap_weight=0.7,
                        base_port=6000, verbose=True, deterministic=True):
    """Run the full evaluation suite for a trained checkpoint.

    Returns dict of all metrics.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {os.path.basename(model_path)}")
    print(f"Config: frame_stack={frame_stack}, heatmap={use_heatmap}")
    if use_heatmap:
        print(f"        heatmap_sigma={heatmap_sigma}, heatmap_weight={heatmap_weight}")
    print(f"deterministic: {deterministic}")
    print(f"Matches per config: {n_matches}")
    print(f"{'='*60}\n")

    model = PPO.load(model_path, device=device)

    # --- Right-side evaluation ---
    results = {"model": os.path.basename(model_path), "n_matches": n_matches,
               "config": {"frame_stack": frame_stack, "use_heatmap": use_heatmap,
                          "heatmap_sigma": heatmap_sigma, "heatmap_weight": heatmap_weight},
               "sides": {}}

    for side_label, play_as_left in [("right", False), ("left", True)]:
        print(f"\n--- {side_label.upper()} SIDE ---")

        env_fn = make_unity_env_fn(
            rank=0 if not play_as_left else 99,
            build_path=BUILD_PATH,
            frame_stack=frame_stack,
            use_ball_tracking=use_heatmap,
            use_heatmap=use_heatmap,
            heatmap_sigma=heatmap_sigma,
            heatmap_weight=heatmap_weight,
            play_as_left=play_as_left,
            base_port=base_port + (1 if play_as_left else 0),
        )
        evaluator = MatchEvaluator(model, env_fn)

        side_results = {}
        for opp in ["no_op", "random", "square"]:
            t0 = time.time()
            r = evaluator.evaluate_vs(opp, n_matches=n_matches, deterministic=deterministic)
            elapsed = time.time() - t0
            side_results[opp] = {
                "win_rate": r["win_rate"],
                "wins": r["wins"],
                "losses": r["losses"],
                "draws": r["draws"],
                "avg_match_len": r["avg_match_len"],
                "avg_point_diff": r["avg_point_diff"],
                "eval_seconds": round(elapsed, 1),
            }
            if verbose:
                wr = r["win_rate"] * 100
                print(f"  vs {opp:8s}: {wr:5.1f}%  ({r['wins']:2d}W {r['losses']:2d}L {r['draws']:2d}D)  "
                      f"pt_diff={r['avg_point_diff']:+5.1f}  len={r['avg_match_len']:.0f}  "
                      f"{elapsed:.0f}s")

        results["sides"][side_label] = side_results

    # --- Spin rate measurement (right side only, vs no_op) ---
    print(f"\n--- SPIN RATE ---")
    spin_env_fn = make_unity_env_fn(
        rank=1, build_path=BUILD_PATH,
        frame_stack=frame_stack, use_ball_tracking=use_heatmap,
        use_heatmap=use_heatmap,
        heatmap_sigma=heatmap_sigma, heatmap_weight=heatmap_weight,
        play_as_left=False, base_port=base_port + 2,
        opponent_policy=lambda obs: np.array([0, 0, 0], dtype=np.int32),
    )
    spin_results = compute_spin_rate(model, spin_env_fn, num_episodes=3)
    results["spin_rate"] = spin_results
    if verbose:
        print(f"  spin_rate: {spin_results['spin_rate']:.4f}  "
              f"({spin_results['spin_steps']}/{spin_results['total_steps']} steps)")

    return results


def print_summary(results):
    """Print compact summary table."""
    print(f"\n{'='*60}")
    print(" SUMMARY")
    print(f"{'='*60}")
    print(f"  Model:     {results['model']}")
    print(f"  Spin rate: {results['spin_rate']['spin_rate']:.4f}")
    for side in ["right", "left"]:
        print(f"\n  {side.upper()} SIDE:")
        for opp in ["no_op", "random", "square"]:
            s = results["sides"][side][opp]
            print(f"    vs {opp:8s}: {s['win_rate']*100:5.1f}%  "
                  f"({s['wins']:2d}W {s['losses']:2d}L {s['draws']:2d}D)  "
                  f"pt_diff={s['avg_point_diff']:+5.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to .zip checkpoint")
    parser.add_argument("--matches", type=int, default=50, help="Matches per opponent")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--frame-stack", type=int, default=4,
                        help="Frame stack used during training (4 for baseline)")
    parser.add_argument("--heatmap", action="store_true",
                        help="Enable heatmap overlay (must match training)")
    parser.add_argument("--heatmap-sigma", type=int, default=5)
    parser.add_argument("--heatmap-weight", type=float, default=0.7)
    parser.add_argument("--no-deterministic", action="store_true",
                        help="Use stochastic sampling (deterministic=False). "
                             "Useful for undertrained models where argmax mode-collapses.")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    results = run_full_evaluation(
        model_path=args.checkpoint,
        n_matches=args.matches,
        device=args.device,
        frame_stack=args.frame_stack,
        use_heatmap=args.heatmap,
        heatmap_sigma=args.heatmap_sigma,
        heatmap_weight=args.heatmap_weight,
        deterministic=not args.no_deterministic,
    )

    print_summary(results)

    # Save results alongside checkpoint
    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out_path = os.path.join(RESULTS_DIR, f"{ckpt_name}_eval.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")
