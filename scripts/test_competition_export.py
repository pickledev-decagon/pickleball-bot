"""
Dry-run test for competition export (AsyncPolicy).

Creates a minimal dummy PPO model, saves it, then tests:
  1. TeamRight loads and returns valid actions
  2. TeamLeft loads and returns valid actions (with flip/mirror)
  3. Both return within <0.01s
  4. Action values are in {0,1,2}
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gymnasium import Env, spaces
from stable_baselines3 import PPO
from agent.competition_export import TeamRight, TeamLeft


class DummyEnv(Env):
    """Minimal dummy env that satisfies SB3 CnnPolicy requirements."""
    def __init__(self, obs_shape=(16, 84, 168)):
        self.observation_space = spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([3, 3, 3])
        self._step = 0

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        self._step += 1
        return self.observation_space.sample(), 0.0, False, False, {}

    def render(self):
        pass

    def close(self):
        pass


def test_export():
    dummy_path = "/tmp/dummy_ppo_model.zip"

    # 1. Create and save a dummy model with (16,84,168) stacked grayscale obs
    print("Creating dummy PPO model (16-frame stack)...")
    env = DummyEnv(obs_shape=(16, 84, 168))
    model = PPO("CnnPolicy", env, verbose=0, device="cpu", n_steps=4,
                policy_kwargs=dict(normalize_images=False))
    model.save(dummy_path)
    print(f"  Saved to {dummy_path}")

    # 2. Test TeamRight with RAW RGB observations (3, 84, 168)
    print("\nTesting TeamRight with raw RGB (3, 84, 168)...")
    right = TeamRight(dummy_path, device="cpu", frame_stack=16)

    for i in range(20):
        # Simulate raw competition observation: (3, 84, 168) channel-first RGB
        raw_obs = np.random.rand(3, 84, 168).astype(np.float32)
        t0 = time.time()
        action = right.policy(raw_obs, reward=0.0)
        elapsed = time.time() - t0

        assert action.shape == (3,), f"Expected shape (3,), got {action.shape}"
        assert action.dtype == np.int32, f"Expected dtype int32, got {action.dtype}"
        assert all(0 <= a <= 2 for a in action), f"Action out of bounds: {action}"
        assert elapsed < 0.01, f"Response too slow: {elapsed*1000:.2f}ms"

        if i == 0:
            print(f"  Call 0 (cold start): {elapsed*1000:.3f}ms — action: {action}")
        elif i == 19:
            print(f"  Call 19 (warm):      {elapsed*1000:.3f}ms — action: {action}")

    print("  ✅ TeamRight passes all checks")

    # 3. Test TeamLeft with RAW RGB observations
    print("\nTesting TeamLeft with raw RGB (3, 84, 168)...")
    left = TeamLeft(dummy_path, device="cpu", frame_stack=16)

    for i in range(20):
        raw_obs = np.random.rand(3, 84, 168).astype(np.float32)
        t0 = time.time()
        action = left.policy(raw_obs, reward=0.0)
        elapsed = time.time() - t0

        assert action.shape == (3,), f"Expected shape (3,), got {action.shape}"
        assert action.dtype == np.int32, f"Expected dtype int32, got {action.dtype}"
        assert all(0 <= a <= 2 for a in action), f"Action out of bounds: {action}"
        assert elapsed < 0.01, f"Response too slow: {elapsed*1000:.2f}ms"

        if i == 0:
            print(f"  Call 0 (cold start): {elapsed*1000:.3f}ms — action: {action}")
        elif i == 19:
            print(f"  Call 19 (warm):      {elapsed*1000:.3f}ms — action: {action}")

    print("  ✅ TeamLeft passes all checks")

    # 4. Test preprocessor directly
    print("\nTesting CompetitionPreprocessor...")
    from agent.competition_export import CompetitionPreprocessor
    prep = CompetitionPreprocessor(frame_stack=16)
    raw = np.random.rand(3, 84, 168).astype(np.float32)
    stacked = prep.process(raw)
    assert stacked.shape == (16, 84, 168), f"Expected (16,84,168), got {stacked.shape}"
    assert stacked.dtype == np.float32, f"Expected float32, got {stacked.dtype}"
    assert 0 <= stacked.min() <= stacked.max() <= 1, "Values out of [0,1] range"
    print(f"  Input:  {raw.shape} -> Output: {stacked.shape}")
    print("  ✅ Preprocessor output shape and dtype correct")

    # 5. Test left-side action mirror logic
    print("\nTesting left-side action mirror logic...")
    from agent.competition_export import AsyncPolicy
    test_cases = [
        (np.array([1, 0, 1]), np.array([1, 2, 1])),
        (np.array([0, 2, 2]), np.array([0, 0, 2])),
        (np.array([2, 1, 0]), np.array([2, 1, 0])),
    ]
    for original, expected in test_cases:
        mirrored = AsyncPolicy._mirror_horizontal_action(original)
        assert np.array_equal(mirrored, expected), f"Mirror failed: {original} -> {mirrored}, expected {expected}"
    print("  ✅ Horizontal action mirror logic correct")

    # 6. Cleanup
    os.remove(dummy_path)
    print(f"\n🎉 All competition export tests passed!")


if __name__ == "__main__":
    test_export()
