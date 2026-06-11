"""
Competition-ready policy export with async inference fallback.

Mirrors the official Competition.py threading model:
  - policy() returns in < 0.01s (always the previous action)
  - Background thread computes next action from current obs
  - If inference is slow, previous action is reused automatically

Also replicates the training wrapper's preprocessing pipeline:
  - Receives raw RGB (3, 84, 168) from competition
  - Transposes, grayscales, resizes, normalizes
  - Maintains frame stack buffer across calls
  - Feeds (stack, 84, 168) to the CNN policy

Usage in competition submission:
    from agent.competition_export import TeamRight, TeamLeft
    team_right = TeamRight("path/to/model.zip", device="cpu")
    action = team_right.policy(observation, reward)  # np.array([v,h,r], dtype=np.int32)
"""

import threading
import collections
import numpy as np
from stable_baselines3 import PPO


class CompetitionPreprocessor:
    """
    Stateful preprocessor that replicates EnhancedUnityWrapper's pipeline
    for competition inference.

    Input:  raw RGB float32 [0,1]  shape (3, 84, 168) or (3, 168, 84)
    Output: stacked grayscale       shape (frame_stack, 84, 168)
    """

    def __init__(self, frame_stack=16, img_size=(84, 168)):
        """
        Args:
            frame_stack: Number of frames to stack (must match training config).
            img_size:    Target (height, width) after resize — (84, 168).
        """
        self.frame_stack = frame_stack
        self.img_size = img_size          # (h, w) = (84, 168)
        self.target_w = img_size[1]       # 168
        self.target_h = img_size[0]       # 84
        self.frames = collections.deque(maxlen=frame_stack)
        self._initialized = False

    def reset(self):
        """Clear frame buffer. Call at start of new match if needed."""
        self.frames.clear()
        self._initialized = False

    def process(self, raw_obs):
        """
        Convert raw competition observation to stacked grayscale frame.

        Args:
            raw_obs: np.ndarray, shape (3, H, W) channel-first RGB, float32 [0,1].

        Returns:
            np.ndarray shape (frame_stack, 84, 168) float32.
        """
        obs = np.array(raw_obs, dtype=np.float32)

        # 1. Ensure shape is (C, H, W)
        if len(obs.shape) != 3:
            raise ValueError(f"Expected 3D observation, got shape {obs.shape}")

        # 2. If channels are not first (e.g. H,W,C), transpose
        #    Heuristic: if first dim <= 4, assume channel-first; else channel-last
        if obs.shape[0] > 4:
            obs = obs.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)

        # 3. Transpose to (H, W, C) for OpenCV
        obs_hwc = obs.transpose(1, 2, 0)

        # 4. Grayscale
        import cv2
        gray = cv2.cvtColor(obs_hwc, cv2.COLOR_RGB2GRAY)

        # 5. Resize to (84, 168) — cv2.resize uses (width, height)
        gray = cv2.resize(gray, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)

        # 6. Normalize to [0, 1]
        gray = gray.astype(np.float32) / 255.0

        # 7. Add channel dim -> (1, 84, 168)
        frame = np.expand_dims(gray, axis=0)

        # 8. Stack
        if not self._initialized:
            # Fill buffer with copies of first frame (matches wrapper reset)
            for _ in range(self.frame_stack):
                self.frames.append(frame)
            self._initialized = True
        else:
            self.frames.append(frame)

        # 9. Concatenate along channel axis -> (stack, 84, 168)
        stacked = np.concatenate(list(self.frames), axis=0)
        return stacked


class AsyncPolicy:
    """
    Async inference wrapper for dPickleBall competition.

    The competition script calls policy(obs, reward) every step with a 0.01s
    deadline. We always return the action computed on the PREVIOUS step,
    while starting background inference for the NEXT step. This guarantees
    < 0.01s response time.
    """

    def __init__(self, model_path, device="cpu", play_as_left=False, frame_stack=16):
        """
        Args:
            model_path: Path to SB3 PPO .zip checkpoint.
            device: "cpu" or "cuda" (competition uses CPU).
            play_as_left: If True, horizontally flip observations before
                          inference and mirror horizontal action afterward,
                          so the same right-side-trained model can play left.
            frame_stack:  Must match the frame_stack used during training.
        """
        self.model = PPO.load(model_path, device=device)
        self.play_as_left = play_as_left
        self.preprocessor = CompetitionPreprocessor(frame_stack=frame_stack)

        # Neutral action = no vertical movement, no horizontal movement, no rotation
        self.current_action = np.array([1, 1, 1], dtype=np.int32)
        self.lock = threading.Lock()
        self.worker = None

    def policy(self, observation, reward):
        """
        Competition API entrypoint.

        Args:
            observation: np.ndarray, raw RGB from Unity, shape (3, H, W).
            reward: scalar (unused but required by API).

        Returns:
            np.ndarray([v, h, r], dtype=np.int32) with values in {0,1,2}
        """
        # Preprocess observation (includes frame stacking)
        try:
            obs = self.preprocessor.process(observation)
        except Exception as e:
            print(f"[AsyncPolicy] Preprocessing error: {e}")
            # On preprocessing failure, still return previous action
            with self.lock:
                return self.current_action.copy()

        # Start background inference for NEXT step if not already running
        if self.worker is None or not self.worker.is_alive():
            self.worker = threading.Thread(
                target=self._infer_thread,
                args=(obs.copy(),),
                daemon=True,
            )
            self.worker.start()

        # Return PREVIOUS action immediately — guaranteed < 0.01s
        with self.lock:
            return self.current_action.copy()

    def _infer_thread(self, obs):
        """Background thread: run model inference."""
        try:
            # If playing left, flip observation so right-side model understands it
            if self.play_as_left:
                obs = self._horizontal_flip_obs(obs)

            # Add batch dimension for SB3
            obs_batch = np.expand_dims(obs, axis=0)
            action, _ = self.model.predict(obs_batch, deterministic=True)
            action = action.astype(np.int32).flatten()[:3]
            action = np.clip(action, 0, 2)

            # If playing left, mirror horizontal action back
            if self.play_as_left:
                action = self._mirror_horizontal_action(action)

            with self.lock:
                self.current_action = action
        except Exception as e:
            # On any inference error, keep previous action (safe fallback)
            print(f"[AsyncPolicy] Inference error: {e}")

    def _horizontal_flip_obs(self, obs):
        """
        Flip observation horizontally (mirror along width axis).
        obs is (stack, H, W) numpy array.
        """
        return np.flip(obs, axis=2).copy()

    @staticmethod
    def _mirror_horizontal_action(action):
        """
        Mirror horizontal action: 0 <-> 2, 1 stays 1.
        action is [vertical, horizontal, rotation].
        """
        mirrored = action.copy()
        h = mirrored[1]
        mirrored[1] = 2 - h  # 0->2, 1->1, 2->0
        return mirrored


class TeamRight(AsyncPolicy):
    """Convenience class for competition submission — right-side agent."""
    def __init__(self, model_path, device="cpu", frame_stack=16):
        super().__init__(model_path, device=device, play_as_left=False, frame_stack=frame_stack)


class TeamLeft(AsyncPolicy):
    """Convenience class for competition submission — left-side agent."""
    def __init__(self, model_path, device="cpu", frame_stack=16):
        super().__init__(model_path, device=device, play_as_left=True, frame_stack=frame_stack)
