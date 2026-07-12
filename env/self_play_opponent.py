"""
Self-play opponent for training.

Uses a PPO checkpoint as the opponent, playing from the LEFT side.
The opponent receives the agent's (right-side) observation, flips it
horizontally to get the left-side perspective, runs inference, and
mirrors the horizontal action back.

This enables progressive self-play: as the agent improves, the opponent
checkpoint is updated to match, creating an ever-stronger training partner.

Usage:
    opp = SelfPlayOpponent("path/to/checkpoint.zip", device="cpu")
    action = opp(obs, ball_pos=(bx, by))  # obs = (4, 84, 168) stacked frames

    # Update to a new checkpoint (e.g., every 50k steps)
    opp.update_model("path/to/newer_checkpoint.zip")
"""

import os
import numpy as np
from stable_baselines3 import PPO
from gymnasium import spaces


class SelfPlayOpponent:
    """
    Opponent that uses a PPO checkpoint to play.

    The observation is horizontally flipped so the model (trained on
    right-side play) sees its paddle on the right side of the frame.
    The action is mirrored ONLY when the opponent plays LEFT side
    (because left-side horizontal directions are reversed).

    opponent_plays_left=True:  flip obs + mirror action (opponent on left)
    opponent_plays_left=False: flip obs only (opponent on right, no mirror)
    """

    def __init__(self, model_path, device="cpu", deterministic=False,
                 opponent_plays_left=True):
        """
        Args:
            model_path: Path to PPO checkpoint (.zip).
            device: Device for inference ("cpu" recommended — don't
                    compete with the training agent for GPU/MPS).
            deterministic: If True, use argmax (deterministic) inference.
                          If False, sample from policy distribution (stochastic).
                          Stochastic is recommended for training variety.
            opponent_plays_left: Which side the opponent plays.
                                True = left side (mirror action).
                                False = right side (no mirror).
        """
        self.device = device
        self.deterministic = deterministic
        self.opponent_plays_left = opponent_plays_left
        self.model_path = model_path
        self.model = PPO.load(model_path, device=device)
        self._fallback_action = np.array([0, 0, 0], dtype=np.int32)
        # Check if model expects Dict obs (MultiInputPolicy) or Box obs (CnnPolicy)
        self._model_expects_dict = isinstance(
            self.model.observation_space, spaces.Dict
        )

    def update_model(self, model_path):
        """Swap to a newer checkpoint (progressive self-play)."""
        if not os.path.exists(model_path):
            print(f"[SelfPlayOpponent] WARNING: checkpoint not found: {model_path}")
            return
        old_model = self.model
        self.model = PPO.load(model_path, device=self.device)
        self.model_path = model_path
        self._model_expects_dict = isinstance(
            self.model.observation_space, spaces.Dict
        )
        # Free old model memory
        del old_model
        print(f"[SelfPlayOpponent] Updated to: {os.path.basename(model_path)}")

    def _horizontal_flip_obs(self, obs):
        """Flip observation horizontally (mirror along width axis).
        obs is (C, H, W) numpy array. Flip along W (axis=2).
        """
        return np.flip(obs, axis=2).copy()

    def _mirror_horizontal_action(self, action):
        """Mirror horizontal action: swap right(1) <-> left(2), keep none(0).
        Action order is [vertical, horizontal, rotation].
        Index 1 = horizontal. Mapping: 0->0, 1->2, 2->1.
        """
        mirrored = action.copy()
        h = mirrored[1]
        mirrored[1] = 0 if h == 0 else 3 - h
        return mirrored

    def __call__(self, obs, ball_pos=None, paddle_pos=None):
        """Run inference: flip obs, predict, mirror action.

        Args:
            obs: Dict {"visual": (C, H, W), "vector": (6,)} from wrapper,
                 or legacy (C, H, W) array for backward compatibility.
            ball_pos: Unused (accepted for backward compatibility).
            paddle_pos: Opponent's paddle position in camera space (x, y).
                        Used to construct the opponent's vector observation.
        Returns:
            np.array([vertical, horizontal, rotation], dtype=int32).
        """
        try:
            if isinstance(obs, dict):
                # --- Dict obs (new format from wrapper) ---
                visual = obs["visual"]
                vector = obs["vector"]

                # Flip visual to opponent's perspective
                flipped_visual = self._horizontal_flip_obs(visual)

                # Mirror vector to opponent's perspective:
                # ball_vx: negate (mirror horizontal direction)
                # ball_vy: same
                # paddle_x/y: replace with opponent's paddle (from paddle_pos kwarg)
                # ball_x: 1.0 - x (mirror)
                # ball_y: same
                flipped_vector = vector.copy()
                flipped_vector[0] = -vector[0]    # ball_vx
                flipped_vector[4] = 1.0 - vector[4]  # ball_x

                # Opponent's paddle position from paddle_pos kwarg
                if paddle_pos is not None and paddle_pos[0] >= 0:
                    if self.opponent_plays_left:
                        flipped_vector[2] = 1.0 - paddle_pos[0]
                    else:
                        flipped_vector[2] = paddle_pos[0]
                    flipped_vector[3] = paddle_pos[1]
                else:
                    flipped_vector[2] = -1.0
                    flipped_vector[3] = -1.0

                if self._model_expects_dict:
                    # New model (MultiInputPolicy): pass full Dict
                    obs_batch = {
                        "visual": np.expand_dims(flipped_visual, axis=0),
                        "vector": np.expand_dims(flipped_vector, axis=0),
                    }
                else:
                    # Old model (CnnPolicy): pass visual only as Box
                    obs_batch = np.expand_dims(flipped_visual, axis=0)
            else:
                # --- Legacy array obs (backward compat) ---
                flipped_obs = self._horizontal_flip_obs(obs)
                obs_batch = np.expand_dims(flipped_obs, axis=0)

            # Predict
            action, _ = self.model.predict(obs_batch, deterministic=self.deterministic)
            action = action.astype(np.int32).flatten()[:3]
            action = np.clip(action, 0, 2)

            # Mirror horizontal action ONLY when opponent plays left side.
            if self.opponent_plays_left:
                action = self._mirror_horizontal_action(action)

            return action
        except Exception as e:
            print(f"[SelfPlayOpponent] Inference error: {e}")
            return self._fallback_action.copy()


def make_self_play_opponent(model_path, device="cpu", deterministic=False,
                            opponent_plays_left=True):
    """Factory function for SelfPlayOpponent."""
    return SelfPlayOpponent(model_path, device=device, deterministic=deterministic,
                            opponent_plays_left=opponent_plays_left)
