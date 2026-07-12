"""Competition entry point — RIGHT-side agent (TeamY).

Official competition interface:
    teamY = TeamY()  # no-arg constructor
    action = teamY.policy(observation, reward)  # returns [v, h, r]

teamY provides actions for 'agent_right_id' (RIGHT side).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from teamX import CompetitionPreprocessor, load_exported_model


class TeamY:
    """Right-side competition agent. No-arg constructor per official interface."""

    def __init__(self):
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model_candidates = [
            os.path.join(model_dir, "exp003_75000_steps.zip"),
            os.path.join(
                model_dir,
                "checkpoints_selfplay_v2",
                "exp003_mixed_opponent",
                "exp003_75000_steps.zip",
            ),
        ]
        model_path = next((path for path in model_candidates if os.path.exists(path)), model_candidates[0])

        self.model = load_exported_model(model_path)
        self.preprocessor = CompetitionPreprocessor(
            frame_stack=4,
            use_heatmap=True,
            heatmap_sigma=5,
            heatmap_weight=0.7,
            play_as_left=False,
        )
        self.current_action = np.array([0, 0, 0], dtype=np.int32)

    def policy(self, observation, reward):
        """Competition API: return action [vertical, horizontal, rotation]."""
        try:
            obs = self.preprocessor.process(observation)
            obs_batch = {
                "visual": np.expand_dims(obs["visual"], axis=0),
                "vector": np.expand_dims(obs["vector"], axis=0),
            }
            action, _ = self.model.predict(obs_batch, deterministic=True)
            action = action.astype(np.int32).flatten()[:3]
            action = np.clip(action, 0, 2)
            self.current_action = action
            return action
        except Exception as e:
            print(f"[TeamY] Error: {e}")
            return self.current_action.copy()
