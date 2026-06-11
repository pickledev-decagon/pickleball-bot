"""
Validate BallTracker + SmoothBallTracker on actual Unity frames.

Usage:
    python scripts/validate_tracker.py --steps 1000

This creates a temporary Unity environment, runs the ball tracker on raw
frames, and reports:
  - Detection rate (% frames where ball was found)
  - Miss rate (% frames where tracker returned (-1,-1))
  - Occlusion recovery rate (miss followed by re-detect within N frames)
  - Sanity checks (ball should generally be within court bounds)
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BUILD_PATH
from env.preprocessing import BallTracker, SmoothBallTracker
from agent.wrapper_vec import make_unity_env_fn


def validate_tracker(n_steps=1000, use_smoothing=True):
    print(f"Creating temporary Unity environment for tracker validation...")
    print(f"  Steps: {n_steps}")
    print(f"  Smoothing: {use_smoothing}")

    # Create a single env with play_as_left=False (standard)
    env_fn = make_unity_env_fn(
        rank=0,
        build_path=BUILD_PATH,
        frame_stack=1,  # Don't stack for tracker validation
        use_ball_tracking=False,  # We'll run tracker manually on raw frames
        use_rgb=False,
    )
    env = env_fn()

    tracker = BallTracker()
    smooth_tracker = SmoothBallTracker() if use_smoothing else None

    detections = 0
    misses = 0
    out_of_bounds = 0
    recoveries = 0
    consecutive_misses = 0
    max_consecutive_misses = 0

    # Monkey-patch _preprocess to capture raw frame before preprocessing
    original_preprocess = env._preprocess
    env._captured_raw = None

    def capturing_preprocess(raw_obs):
        # Save the raw obs before any processing
        if raw_obs is not None:
            if raw_obs.shape[0] == 3:
                # Channel-first (C,H,W) -> (H,W,C)
                env._captured_raw = raw_obs.transpose(1, 2, 0)
            else:
                env._captured_raw = raw_obs
        return original_preprocess(raw_obs)

    env._preprocess = capturing_preprocess

    # Also capture from reset
    original_reset = env.reset
    env._last_raw_reset = None

    def capturing_reset(*, seed=None, options=None):
        result = original_reset(seed=seed, options=options)
        env._last_raw_reset = env._captured_raw
        return result

    env.reset = capturing_reset

    obs, info = env.reset()

    # Use captured frame from reset for first detection
    raw_frame = env._last_raw_reset if env._last_raw_reset is not None else env._captured_raw

    for step in range(n_steps):
        # Process the captured raw frame
        if raw_frame is not None:
            # Convert to uint8 for OpenCV
            if raw_frame.max() <= 1.0:
                frame_uint8 = (raw_frame * 255).astype(np.uint8)
            else:
                frame_uint8 = raw_frame.astype(np.uint8)

            detected = tracker.detect(frame_uint8)
            if smooth_tracker is not None:
                smoothed = smooth_tracker.update(detected)
                pos = smoothed
            else:
                pos = detected

            if pos[0] < 0:
                misses += 1
                consecutive_misses += 1
                max_consecutive_misses = max(max_consecutive_misses, consecutive_misses)
            else:
                detections += 1
                if consecutive_misses > 0 and consecutive_misses <= 5:
                    recoveries += 1
                consecutive_misses = 0

                # Sanity check: ball should be roughly in court (0,1) range
                if not (0 <= pos[0] <= 1 and 0 <= pos[1] <= 1):
                    out_of_bounds += 1

        # Step the environment
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        # Grab the raw frame captured during this step's _preprocess call
        raw_frame = env._captured_raw

        if done or truncated:
            obs, info = env.reset()
            raw_frame = env._last_raw_reset

    env.close()

    total = detections + misses
    if total == 0:
        print("No valid frames processed.")
        return

    print(f"\n{'='*50}")
    print("Tracker Validation Results")
    print(f"{'='*50}")
    print(f"  Total frames processed: {total}")
    print(f"  Detections:             {detections} ({detections/total*100:.1f}%)")
    print(f"  Misses:                 {misses} ({misses/total*100:.1f}%)")
    print(f"  Max consecutive misses: {max_consecutive_misses}")
    print(f"  Quick recoveries (<=5): {recoveries}")
    print(f"  Out-of-bounds detections: {out_of_bounds}")
    print(f"{'='*50}")

    # Assertions / warnings
    if misses / total > 0.10:
        print("\n⚠️  WARNING: Miss rate > 10%. Heatmap obs may be unreliable.")
    else:
        print("\n✅ Tracker miss rate acceptable (< 10%)")

    if max_consecutive_misses > 20:
        print("⚠️  WARNING: Max consecutive misses > 20. Occlusion handling is weak.")
    else:
        print("✅ Consecutive miss streak acceptable")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate ball tracker on real Unity frames")
    parser.add_argument("--steps", type=int, default=1000, help="Number of env steps to test")
    parser.add_argument("--no-smoothing", action="store_true", help="Disable SmoothBallTracker")
    args = parser.parse_args()

    validate_tracker(n_steps=args.steps, use_smoothing=not args.no_smoothing)
