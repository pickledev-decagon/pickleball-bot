"""
Showcase: watch the Phase 5 500k agent play live in the Unity window.

Opens the dPickleball Unity build and runs the agent against a chosen opponent.
The Unity game window will appear — sit back and watch your agent play!

Usage:
    python showcase.py                          # self-play (default)
    python showcase.py --opponent selfplay      # agent vs itself
    python showcase.py --opponent square        # vs square-motion opponent
    python showcase.py --opponent ball_tracking # vs ball-tracking opponent
    python showcase.py --opponent no_op         # vs stationary opponent
    python showcase.py --opponent random        # vs random actions
    python showcase.py --left                   # play as left side
    python showcase.py --speed 2                # faster stepping (skip frames)
    python showcase.py --matches 3              # play 3 matches then exit
"""

import argparse
import os
import sys
import time
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from agent.wrapper_vec import make_unity_env_fn
from config import BUILD_PATH, check_build_path


def make_square_opponent():
    """Square-motion opponent (curriculum Stage 2)."""
    step_counter = [0]
    def _square_policy(obs):
        side = (step_counter[0] // 30) % 4
        dirs = [
            np.array([1, 0, 0], dtype=np.int32),
            np.array([0, 1, 0], dtype=np.int32),
            np.array([2, 0, 0], dtype=np.int32),
            np.array([0, 2, 0], dtype=np.int32),
        ]
        step_counter[0] += 1
        return dirs[side]
    return _square_policy


OPPONENTS = {
    "no_op": lambda obs: np.array([0, 0, 0], dtype=np.int32),
    "random": lambda obs: np.random.randint(0, 3, size=3, dtype=np.int32),
    "square": make_square_opponent(),
    "ball_tracking": None,  # Lazy-imported below
}


def get_opponent(name):
    if name == "ball_tracking":
        from env.ball_tracking_opponent import BallTrackingOpponent
        return BallTrackingOpponent(hit_rate=0.6, noise_rate=0.2, reaction_delay=3, seed=42)
    return OPPONENTS[name]


def make_selfplay_opponent(model, device="cpu"):
    """Self-play opponent using the same PPO model."""
    def _selfplay_policy(obs):
        action, _ = model.predict(obs, deterministic=False)
        return np.clip(action.astype(np.int32), 0, 2)
    return _selfplay_policy


def main():
    parser = argparse.ArgumentParser(description="Watch the Phase 5 500k agent play live")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints_selfplay_v2/phase5_selfplay_20260628_050245/phase5_500000_steps.zip",
                        help="Path to checkpoint zip")
    parser.add_argument("--opponent", type=str, default="selfplay",
                        choices=list(OPPONENTS.keys()) + ["ball_tracking", "selfplay"],
                        help="Opponent type to play against")
    parser.add_argument("--left", action="store_true",
                        help="Play as left side (default: right)")
    parser.add_argument("--matches", type=int, default=0,
                        help="Number of matches to play (0 = infinite)")
    parser.add_argument("--port", type=int, default=6000,
                        help="Base port for Unity worker")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for model inference")
    args = parser.parse_args()

    if not check_build_path():
        print("Unity build not found. Set DPICKLEBALL_BUILD_PATH.")
        sys.exit(1)

    # Resolve checkpoint path
    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ckpt_path)
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"PICKLEBALL SHOWCASE — Phase 5 500k")
    print(f"{'='*60}")
    print(f"  Checkpoint:  {os.path.basename(ckpt_path)}")
    print(f"  Side:        {'LEFT' if args.left else 'RIGHT'}")
    print(f"  Opponent:    {args.opponent}")
    print(f"  Matches:     {'∞ (Ctrl+C to stop)' if args.matches == 0 else args.matches}")
    print(f"{'='*60}")
    print(f"\n  ⚡ The Unity game window should appear now.")
    print(f"  Watch your agent play pickleball in real time!\n")

    # Load model
    model = PPO.load(ckpt_path, device=args.device)

    # Create a single environment (one Unity window)
    side_label = "left" if args.left else "right"
    env_fn = make_unity_env_fn(
        rank=0,
        build_path=BUILD_PATH,
        frame_stack=4,
        use_ball_tracking=True,
        use_heatmap=True,
        heatmap_sigma=5,
        heatmap_weight=0.7,
        play_as_left=args.left,
        base_port=args.port,
    )
    env = env_fn()
    env.apply_reward_shaping = False  # No training rewards during showcase

    # Set opponent
    if args.opponent == "selfplay":
        env.opponent_policy = make_selfplay_opponent(model, args.device)
    else:
        env.opponent_policy = get_opponent(args.opponent)

    match_count = 0
    total_matches = args.matches or float('inf')

    try:
        while match_count < total_matches:
            match_count += 1
            obs, info = env.reset()
            my_score = 0
            opp_score = 0
            step = 0
            match_start = time.time()

            print(f"\n--- Match {match_count} starting ---")

            while True:
                # Agent action
                action, _ = model.predict(obs, deterministic=False)
                action = np.clip(action.astype(np.int32), 0, 2)

                obs, reward, done, truncated, info = env.step(action)
                step += 1

                # Track score from info
                if "my_score" in info and "opp_score" in info:
                    my_score = info["my_score"]
                    opp_score = info["opp_score"]
                elif reward > 0.8:
                    my_score += 1
                elif reward < -0.8:
                    opp_score += 1

                # Print progress every 500 steps
                if step % 500 == 0:
                    elapsed = time.time() - match_start
                    print(f"  Match {match_count}: step {step:5d} | {side_label} {my_score} - {opp_score} opponent ({elapsed:.0f}s)")

                if done or truncated or my_score >= 21 or opp_score >= 21:
                    elapsed = time.time() - match_start
                    result = "WON" if my_score > opp_score else "LOST" if my_score < opp_score else "DRAW"
                    print(f"  Match {match_count} {result}!  {my_score} - {opp_score}  ({elapsed:.0f}s, {step} steps)")
                    break

    except KeyboardInterrupt:
        print(f"\n\nShowcase stopped after {match_count} matches.")
    finally:
        env.close()
        print("Unity environment closed.")


if __name__ == "__main__":
    main()
