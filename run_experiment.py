"""
Experiment harness for dPickleBall competition.
Run a named experiment config for N steps, output JSON results.

Usage:
    python run_experiment.py --config baseline --steps 10000 --envs 2
    python run_experiment.py --config entropy_0.05
    python run_experiment.py --list

Output: results/<config_name>_<timestamp>.json
        results/<config_name>_<timestamp>/   (TensorBoard logs)
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BUILD_PATH, RESULTS_DIR

from agent.experiment_configs import resolve_config, EXPERIMENTS
from agent.wrapper_vec import create_vec_env


def run_experiment(config_name, total_steps=10_000, num_envs=2):
    config = resolve_config(config_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    run_tag = f"{config_name}_{timestamp}"
    log_dir = os.path.join(results_dir, run_tag)

    print(f"\n{'='*60}")
    print(f"Experiment: {config['name']}")
    print(f"Config:     {config_name}")
    print(f"Steps:      {total_steps}")
    print(f"Envs:       {num_envs}")
    print(f"Desc:       {config['description']}")
    print(f"{'='*60}\n")

    env_kw = config.get("env_kwargs", {})
    reward_kw = config.get("reward_kwargs", {})

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

    policy_kwargs = {
        "normalize_images": False,
    }
    ppo_kwargs = {
        "learning_rate": 2.5e-4,
        "n_steps": 2048 // num_envs,
        "batch_size": 128,
        "n_epochs": 4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "device": "cpu",
        "verbose": 1,
        "policy_kwargs": policy_kwargs,
    }
    ppo_kwargs.update(config.get("ppo_kwargs", {}))

    model = PPO("CnnPolicy", vec_env, **ppo_kwargs, tensorboard_log=None)

    print(f"Starting training...\n")
    start = time.time()
    model.learn(
        total_timesteps=total_steps,
        tb_log_name=config_name,
        reset_num_timesteps=True,
    )
    elapsed = time.time() - start

    vec_env.close()

    metrics = {}
    for key in ["train/entropy_loss", "train/policy_gradient_loss", "train/value_loss", "train/approx_kl"]:
        val = model.logger.name_to_value.get(key, None)
        if val is not None:
            metrics[key] = float(val)

    result = {
        "experiment": config_name,
        "description": config["description"],
        "timestamp": timestamp,
        "total_steps": total_steps,
        "num_envs": num_envs,
        "elapsed_seconds": round(elapsed, 1),
        "steps_per_second": round(total_steps / elapsed, 1) if elapsed > 0 else 0,
        **metrics,
    }

    result_path = os.path.join(results_dir, f"{run_tag}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print(" Results:")
    for k, v in result.items():
        if k not in ("experiment", "description", "timestamp"):
            print(f"  {k}: {v}")
    print(f"  saved to: {result_path}")
    print(f"{'='*60}\n")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dPickleBall experiment runner")
    parser.add_argument("--config", type=str, default="baseline")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--envs", type=int, default=2)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("Available experiments:")
        for name, cfg in sorted(EXPERIMENTS.items()):
            resolved = resolve_config(name)
            print(f"  {name:20s} — {resolved['description']}")
        sys.exit(0)

    run_experiment(args.config, args.steps, args.envs)
