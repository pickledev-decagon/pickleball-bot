#!/usr/bin/env python3
"""
Random hyperparameter sweep for dPickleBall PPO.

Reads a JSON sweep config, samples N random trials, trains each,
and collates results into summary.json + best.json.

Usage:
    python scripts/grid_search.py sweep_configs/phase3_default.json
    python scripts/grid_search.py sweep_configs/phase3_focused.json --output results/sweeps
"""

import os
import sys
import json
import math
import random
import time
import argparse
import csv
import subprocess
import signal
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def kill_unity():
    """Kill any lingering Unity process to free the communication port."""
    subprocess.run(
        ["pkill", "-9", "-f", "Mac.app/Contents/MacOS/dp"],
        capture_output=True, timeout=10
    )
    try:
        res = subprocess.run(
            ["lsof", "-ti:5005-5012"], capture_output=True, text=True, timeout=5
        )
        for pid in res.stdout.strip().split("\n"):
            if pid.strip():
                os.kill(int(pid.strip()), signal.SIGKILL)
    except Exception:
        pass
    time.sleep(2)


from config import BUILD_PATH
from stable_baselines3 import PPO
from agent.experiment_configs import resolve_config
from agent.wrapper_vec import create_vec_env


def sample_param(spec):
    """Sample a single parameter value from its distribution spec."""
    ptype = spec.get("type", "choice")
    if ptype == "choice":
        return random.choice(spec["values"])
    elif ptype == "uniform":
        return random.uniform(spec["min"], spec["max"])
    elif ptype == "loguniform":
        return 10 ** random.uniform(
            math.log10(spec["min"]), math.log10(spec["max"])
        )
    else:
        raise ValueError(f"Unknown param type: {ptype}")


def set_nested(config, dot_key, value):
    """
    Set a nested dict value using dot notation.
    E.g. 'ppo_kwargs.ent_coef' -> config['ppo_kwargs']['ent_coef'] = value
    """
    keys = dot_key.split(".")
    d = config
    for k in keys[:-1]:
        if k not in d:
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def run_trial(trial_id, resolved_config, steps, num_envs):
    """
    Run a single training trial and return metrics dict.
    """
    env_kw = resolved_config.get("env_kwargs", {})
    reward_kw = resolved_config.get("reward_kwargs", {})

    vec_env = create_vec_env(
        build_path=BUILD_PATH,
        num_envs=num_envs,
        frame_stack=env_kw.get("frame_stack", 4),
        use_ball_tracking=True,
        use_rgb=env_kw.get("use_rgb", False),
        use_frame_diff=env_kw.get("use_frame_diff", False),
        use_heatmap=env_kw.get("use_heatmap", False),
        reward_config=reward_kw if reward_kw else None,
    )

    # Default PPO kwargs (same as run_experiment.py)
    ppo_kwargs = {
        "learning_rate": 2.5e-4,
        "n_steps": max(2048 // num_envs, 64),
        "batch_size": 128,
        "n_epochs": 4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "device": "cpu",
        "verbose": 0,  # quiet during sweep
        "policy_kwargs": {"normalize_images": False},
    }

    # Apply config overrides
    for key, value in resolved_config.get("ppo_kwargs", {}).items():
        ppo_kwargs[key] = value

    # Safety: batch_size must divide into total rollout buffer size
    buffer_size = ppo_kwargs["n_steps"] * num_envs
    if ppo_kwargs["batch_size"] > buffer_size:
        ppo_kwargs["batch_size"] = buffer_size
        print(f"  [trial {trial_id}] Clamped batch_size to {buffer_size}")

    model = PPO("CnnPolicy", vec_env, **ppo_kwargs, tensorboard_log=None)

    start = time.time()
    model.learn(total_timesteps=steps, reset_num_timesteps=True)
    elapsed = time.time() - start

    vec_env.close()

    # Extract metrics from SB3 logger
    metrics = {}
    metric_keys = [
        "train/entropy_loss",
        "train/policy_gradient_loss",
        "train/value_loss",
        "train/approx_kl",
        "train/explained_variance",
        "train/clip_range",
    ]
    for key in metric_keys:
        val = model.logger.name_to_value.get(key, None)
        if val is not None:
            metrics[key] = float(val)

    result = {
        "trial_id": trial_id,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "total_steps": steps,
        "num_envs": num_envs,
        "elapsed_seconds": round(elapsed, 1),
        "steps_per_second": round(steps / elapsed, 1) if elapsed > 0 else 0,
        "config": resolved_config,
        **metrics,
    }
    return result


def collate_results(output_dir, results):
    """Write summary.json, best.json, and a CSV summary."""
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    # CSV
    csv_path = os.path.join(output_dir, "summary.csv")
    valid = [r for r in results if "error" not in r]
    if valid:
        keys = ["trial_id", "elapsed_seconds", "steps_per_second",
                "train/entropy_loss", "train/value_loss",
                "train/approx_kl", "train/explained_variance"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for r in valid:
                row = {k: r.get(k, "") for k in keys}
                row["trial_id"] = r["trial_id"]
                writer.writerow(row)
        print(f"  CSV: {csv_path}")

    # Best by explained_variance (higher = critic learning signal)
    if valid:
        best = max(
            valid,
            key=lambda r: r.get("train/explained_variance", -999),
        )
        best_path = os.path.join(output_dir, "best.json")
        with open(best_path, "w") as f:
            json.dump(best, f, indent=2)
        print(
            f"\nBest trial: {best['trial_id']} "
            f"— explained_variance={best.get('train/explained_variance')}, "
            f"value_loss={best.get('train/value_loss')}, "
            f"fps={best.get('steps_per_second')}"
        )
        return best
    return None


def main():
    parser = argparse.ArgumentParser(description="dPickleBall hyperparameter sweep")
    parser.add_argument("config", type=str, help="Path to sweep JSON config")
    parser.add_argument(
        "--output", type=str, default="results/sweeps",
        help="Output directory for sweep results",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        sweep = json.load(f)

    name = sweep.get("name", "sweep")
    base_config_name = sweep.get("base_config", "baseline")
    n_trials = sweep.get("n_trials", 30)
    steps_per_trial = sweep.get("steps_per_trial", 20_000)
    num_envs = sweep.get("envs", 2)

    output_dir = os.path.join(args.output, name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Sweep: {name}")
    print(f"  Base config:   {base_config_name}")
    print(f"  Trials:        {n_trials}")
    print(f"  Steps/trial:   {steps_per_trial}")
    print(f"  Envs:          {num_envs}")
    print(f"  Output dir:    {output_dir}")
    print(f"  Params:        {list(sweep.get('params', {}).keys())}")

    # Kill any stale Unity before starting trials
    kill_unity()

    results = []
    for i in range(n_trials):
        print(f"\n--- Trial {i + 1}/{n_trials} ---")
        base = resolve_config(base_config_name)
        trial_config = dict(base)

        # Apply env_kwargs overrides from sweep config (e.g. frame_stack)
        sweep_env_kw = sweep.get("env_kwargs", {})
        if sweep_env_kw:
            trial_config.setdefault("env_kwargs", {}).update(sweep_env_kw)

        # Apply reward_kwargs overrides from sweep config (e.g. hit_bonus)
        sweep_reward_kw = sweep.get("reward_kwargs", {})
        if sweep_reward_kw:
            trial_config.setdefault("reward_kwargs", {}).update(sweep_reward_kw)

        # Sample and apply sweep parameters
        sampled = {}
        for dot_key, spec in sweep.get("params", {}).items():
            value = sample_param(spec)
            sampled[dot_key] = value
            set_nested(trial_config, dot_key, value)
            print(f"  {dot_key:30s} = {value}")

        try:
            result = run_trial(i, trial_config, steps_per_trial, num_envs)
            result["sampled_params"] = sampled
            results.append(result)

            out_path = os.path.join(output_dir, f"trial_{i:03d}.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(
                f"  Saved: {out_path} | "
                f"fps={result.get('steps_per_second')} | "
                f"expl_var={result.get('train/explained_variance')}"
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "trial_id": i,
                "error": str(e),
                "sampled_params": sampled,
                "config": trial_config,
            })
        finally:
            # Kill Unity to free socket for next trial
            kill_unity()

    collate_results(output_dir, results)
    print(f"\nSummary: {os.path.join(output_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
