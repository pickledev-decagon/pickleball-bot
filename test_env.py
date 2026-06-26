"""
End-to-end pipeline validation script.

Tests that the complete RL pipeline works correctly:
  Unity env creation → wrapper preprocessing → PPO model inference

Also computes quantitative metrics (spin rate) used to measure
whether bug fixes are actually working.

Run:  python test_env.py
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from config import BUILD_PATH
from agent.wrapper_vec import make_unity_env_fn


def make_vec_env():
    """Create a single-vectorised Unity env with minimal settings."""
    env_fn = make_unity_env_fn(
        rank=0,
        build_path=BUILD_PATH,
        frame_stack=4,
        use_ball_tracking=False,
        use_heatmap=False,
        base_port=5010,
        reward_config=None,
        opponent_policy=None,
    )
    return DummyVecEnv([env_fn])


def test_pipeline():
    """Run the full pipeline end-to-end and assert every step."""
    env = make_vec_env()

    # 1. Reset → observe
    obs = env.reset()                          # (1, 4, 84, 168)
    assert obs.shape == (1, 4, 84, 168), \
        f"Obs shape: expected (1, 4, 84, 168), got {obs.shape}"
    assert obs.dtype == np.float32, \
        f"Obs dtype: expected float32, got {obs.dtype}"
    assert obs.min() >= 0.0 and obs.max() <= 1.0, \
        f"Obs range: expected [0,1], got [{obs.min():.3f}, {obs.max():.3f}]"
    print(f"  ✓ reset: obs {obs.shape} float32 "
          f"[{obs.min():.3f}, {obs.max():.3f}]")

    # 2. Create a fresh PPO model and run inference once
    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs={"normalize_images": False},
        device="cpu",
        verbose=0,
    )
    single_obs = obs[0]                        # (4, 84, 168) for predict
    action, _ = model.predict(single_obs, deterministic=True)
    assert action.shape == (3,), \
        f"Action shape: expected (3,), got {action.shape}"
    assert action.dtype in (np.int32, np.int64), \
        f"Action dtype: expected int32/int64, got {action.dtype}"
    assert np.all((0 <= action) & (action <= 2)), \
        f"Action values: expected [0,2], got {action}"
    print(f"  ✓ predict: {action} ({action.dtype})")

    # 3. Step the env (action needs batch dim)
    vec_action = action.reshape(1, 3)
    obs2, reward, done, info = env.step(vec_action)
    assert obs2.shape == (1, 4, 84, 168), \
        f"Step obs shape: expected (1, 4, 84, 168), got {obs2.shape}"
    print(f"  ✓ step: reward={reward[0]:.4f}  done={done[0]}")

    # 4. Compute spin metric
    results = compute_spin_rate(model, env, num_episodes=1)
    print(f"  ✓ spin_rate={results['spin_rate']:.4f}  "
          f"contact_rate={results['paddle_contact_rate']:.4f}")

    env.close()
    print("\n✅ ALL PIPELINE TESTS PASSED")
    return results


def compute_spin_rate(model, env, num_episodes=1, max_steps_per_ep=100):
    """Quantify paddle spinning over several evaluation episodes.

    Returns
    -------
    dict
        spin_rate:  fraction of steps where rotation action != 1
        paddle_contact_rate: fraction of steps where paddle moves without spin
    """
    spin_count = 0
    contact_count = 0
    total_steps = 0

    for ep in range(num_episodes):
        obs = env.reset()
        done = [False]
        for _ in range(max_steps_per_ep):
            if done[0]:
                break
            action, _ = model.predict(obs[0], deterministic=True)
            if action[2] != 1:               # rotation = spin
                spin_count += 1
            if action[2] == 1 and action[0] != 1:   # moving = contact
                contact_count += 1
            total_steps += 1
            obs, reward, done, info = env.step(action.reshape(1, 3))

    total = max(total_steps, 1)
    return {
        "spin_rate": spin_count / total,
        "paddle_contact_rate": contact_count / total,
    }


if __name__ == "__main__":
    test_pipeline()
