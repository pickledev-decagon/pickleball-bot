"""
Competition entry — LEFT-side agent (TeamX).

Self-contained: all preprocessing embedded, no imports from project modules.
Loads model checkpoint from same directory as this file.

Official competition interface:
    from teamX import TeamX
    teamX = TeamX()  # no-arg constructor
    action = teamX.policy(observation, reward)  # returns np.array([v, h, r])

teamX provides actions for 'agent_left_id' (LEFT side).
"""

import os
import collections
import numpy as np
import cv2
from stable_baselines3 import PPO

try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces


def _ppo_custom_objects():
    """Replace pickled spaces when loading across OS/SB3/Gym versions."""
    return {
        "observation_space": spaces.Dict({
            "visual": spaces.Box(low=0.0, high=1.0, shape=(4, 84, 168), dtype=np.float32),
            "vector": spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
        }),
        "action_space": spaces.MultiDiscrete([3, 3, 3]),
    }


def load_exported_model(model_path):
    return PPO.load(model_path, device="cpu", custom_objects=_ppo_custom_objects())


# ============================================================================
# Ball Tracking (RGB2HSV — yellow ball)
# ============================================================================

class BallTracker:
    """Detect yellow pickleball via HSV color masking."""

    def __init__(self):
        self.lower_yellow = np.array([20, 80, 80], dtype=np.uint8)
        self.upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
        self.min_ball_area = 5
        self.max_ball_area = 200

    def detect(self, frame_rgb):
        """Return (x, y) normalized [0,1] ball position, or (-1, -1) if not found."""
        if frame_rgb is None or frame_rgb.size == 0:
            return (-1.0, -1.0)

        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return (-1.0, -1.0)

        valid = [c for c in contours if self.min_ball_area < cv2.contourArea(c) < self.max_ball_area]
        if not valid:
            return (-1.0, -1.0)

        largest = max(valid, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return (-1.0, -1.0)

        h, w = frame_rgb.shape[:2]
        cx = M["m10"] / M["m00"] / w
        cy = M["m01"] / M["m00"] / h
        return (cx, cy)


class SmoothBallTracker:
    """Temporal smoothing for ball position — handles brief occlusions."""

    def __init__(self, alpha=0.7):
        self.last_pos = None
        self.velocity = np.array([0.0, 0.0])
        self.alpha = alpha

    def update(self, detected_pos):
        if detected_pos[0] < 0:
            if self.last_pos is not None:
                predicted = self.last_pos + self.velocity
                predicted = np.clip(predicted, 0.0, 1.0)
                self.last_pos = predicted
                return (predicted[0], predicted[1])
            return (-1.0, -1.0)

        pos = np.array(detected_pos, dtype=np.float32)
        if self.last_pos is not None:
            raw_vel = pos - self.last_pos
            self.velocity = self.alpha * self.velocity + (1 - self.alpha) * raw_vel
        self.last_pos = pos
        return (pos[0], pos[1])


# ============================================================================
# Paddle Detection (BGR2HSV — red paddles appear as blue in HSV)
# ============================================================================

def detect_paddles(frame_rgb):
    """Detect paddle positions via BGR2HSV color masking.

    Returns list of (x, y) tuples for 2 largest paddle-like contours.
    """
    lower_color = np.array([100, 50, 50], dtype=np.uint8)
    upper_color = np.array([130, 255, 255], dtype=np.uint8)

    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    positions = []
    for c in contours:
        area = cv2.contourArea(c)
        if 20 < area < 500:
            x, y, w, h = cv2.boundingRect(c)
            cx = (x + w / 2) / frame_rgb.shape[1]
            cy = (y + h / 2) / frame_rgb.shape[0]
            positions.append((cx, cy, area))

    positions.sort(key=lambda p: p[2], reverse=True)
    results = [(-1.0, -1.0), (-1.0, -1.0)]
    for i, (cx, cy, _) in enumerate(positions[:2]):
        results[i] = (cx, cy)
    return results


# ============================================================================
# Preprocessor (replicates training wrapper pipeline)
# ============================================================================

class CompetitionPreprocessor:
    """Convert raw RGB observation to Dict obs for MultiInputPolicy.

    Input:  raw RGB (3, 84, 168) float32 [0,1]
    Output: {"visual": (4, 84, 168) float32, "vector": (6,) float32}
    """

    def __init__(self, frame_stack=4, use_heatmap=True,
                 heatmap_sigma=5, heatmap_weight=0.7, play_as_left=False):
        self.frame_stack = frame_stack
        self.target_h = 84
        self.target_w = 168
        self.use_heatmap = use_heatmap
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_weight = heatmap_weight
        self.frames = collections.deque(maxlen=frame_stack)
        self._initialized = False
        self.ball_tracker = BallTracker() if use_heatmap else None
        self.smooth_tracker = SmoothBallTracker() if use_heatmap else None
        self._last_ball_pos_for_vel = (-1.0, -1.0)
        self.play_as_left = play_as_left

    def reset(self):
        self.frames.clear()
        self._initialized = False
        self._last_ball_pos_for_vel = (-1.0, -1.0)
        if self.smooth_tracker is not None:
            self.smooth_tracker = SmoothBallTracker()

    def _compute_vector_obs(self, ball_pos, paddle_pos):
        """Compute 6-dim vector: [ball_vx, ball_vy, paddle_x, paddle_y, ball_x, ball_y]."""
        if self._last_ball_pos_for_vel[0] >= 0 and ball_pos[0] >= 0:
            ball_vx = np.clip((ball_pos[0] - self._last_ball_pos_for_vel[0]) * 10, -1.0, 1.0)
            ball_vy = np.clip((ball_pos[1] - self._last_ball_pos_for_vel[1]) * 10, -1.0, 1.0)
        else:
            ball_vx = 0.0
            ball_vy = 0.0
        self._last_ball_pos_for_vel = ball_pos

        return np.array(
            [ball_vx, ball_vy, paddle_pos[0], paddle_pos[1], ball_pos[0], ball_pos[1]],
            dtype=np.float32,
        )

    def _render_heatmap(self, gray_frame, ball_pos):
        """Overlay Gaussian heatmap at ball position on grayscale frame."""
        if ball_pos[0] < 0:
            return gray_frame
        h, w = gray_frame.shape[1], gray_frame.shape[2]
        gx = int(np.clip(ball_pos[0] * w, 0, w - 1))
        gy = int(np.clip(ball_pos[1] * h, 0, h - 1))
        gaussian = np.zeros((h, w), dtype=np.float32)

        sigma = self.heatmap_sigma
        kernel_size = 6 * sigma + 1
        kx = cv2.getGaussianKernel(kernel_size, sigma)
        ky = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = kx @ ky.T
        kernel = kernel / kernel.max()

        x1 = max(0, gx - kernel_size // 2)
        x2 = min(w, gx + kernel_size // 2 + 1)
        y1 = max(0, gy - kernel_size // 2)
        y2 = min(h, gy + kernel_size // 2 + 1)
        kx1 = kernel_size // 2 - (gx - x1)
        kx2 = kernel_size // 2 + (x2 - gx)
        ky1 = kernel_size // 2 - (gy - y1)
        ky2 = kernel_size // 2 + (y2 - gy)
        gaussian[y1:y2, x1:x2] = kernel[ky1:ky2, kx1:kx2]
        heatmap = np.expand_dims(gaussian, axis=0)
        return np.clip(gray_frame + heatmap * self.heatmap_weight, 0.0, 1.0)

    def process(self, raw_obs):
        """Convert raw competition observation to Dict obs."""
        obs = np.array(raw_obs, dtype=np.float32)

        if len(obs.shape) != 3:
            raise ValueError(f"Expected 3D observation, got shape {obs.shape}")

        # Transpose if channels not first
        if obs.shape[0] > 4:
            obs = obs.transpose(2, 0, 1)

        # To (H, W, C) for OpenCV
        obs_hwc = obs.transpose(1, 2, 0)
        frame_uint8 = (obs_hwc * 255).astype(np.uint8)

        # Ball tracking
        ball_pos = (-1.0, -1.0)
        if self.use_heatmap and self.ball_tracker is not None:
            detected = self.ball_tracker.detect(frame_uint8)
            if self.smooth_tracker is not None:
                ball_pos = self.smooth_tracker.update(detected)
            else:
                ball_pos = detected

        # Paddle detection
        paddle_positions = detect_paddles(frame_uint8)
        valid_paddles = [p for p in paddle_positions if p[0] >= 0]
        paddle_pos = (-1.0, -1.0)
        if len(valid_paddles) >= 1:
            if self.play_as_left:
                ours = min(valid_paddles, key=lambda p: p[0])
                px, py = ours
                paddle_pos = (1.0 - px, py) if px >= 0 else (-1.0, -1.0)
            else:
                ours = max(valid_paddles, key=lambda p: p[0])
                paddle_pos = ours

        # Mirror ball_pos for left-side play
        if self.play_as_left and ball_pos[0] >= 0:
            ball_pos = (1.0 - ball_pos[0], ball_pos[1])

        # Mirror the visual before grayscale for left-side play. This matches
        # the training wrapper's left-side preprocessing order.
        model_hwc = cv2.flip(obs_hwc, 1) if self.play_as_left else obs_hwc

        # Grayscale
        gray = cv2.cvtColor(model_hwc, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)
        gray = gray.astype(np.float32)
        frame = np.expand_dims(gray, axis=0)

        # Heatmap overlay. For left-side play, ball_pos is already mirrored to
        # the flipped visual coordinate frame.
        if self.use_heatmap:
            frame = self._render_heatmap(frame, ball_pos)

        # Frame stacking
        if not self._initialized:
            for _ in range(self.frame_stack):
                self.frames.append(frame)
            self._initialized = True
        else:
            self.frames.append(frame)

        # Vector obs
        vector_obs = self._compute_vector_obs(ball_pos, paddle_pos)

        visual_obs = np.concatenate(list(self.frames), axis=0)
        return {"visual": visual_obs, "vector": vector_obs}


# ============================================================================
# TeamX — LEFT-side agent
# ============================================================================

class TeamX:
    """Left-side competition agent. No-arg constructor per official interface.

    teamX provides actions for 'agent_left_id' (LEFT side).
    Uses the same model as TeamY but mirrors observations and actions
    so the right-side-trained model can play left.
    """

    def __init__(self):
        self.play_as_left = True  # TeamX = LEFT side
        self.frame_stack = 4
        self.use_heatmap = True
        self.heatmap_sigma = 5
        self.heatmap_weight = 0.7

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
            frame_stack=self.frame_stack,
            use_heatmap=self.use_heatmap,
            heatmap_sigma=self.heatmap_sigma,
            heatmap_weight=self.heatmap_weight,
            play_as_left=self.play_as_left,
        )

        # Fallback action (neutral = no movement, no rotation)
        self.current_action = np.array([0, 0, 0], dtype=np.int32)

    def policy(self, observation, reward):
        """Competition API: return action [vertical, horizontal, rotation].

        Args:
            observation: np.ndarray, raw RGB from Unity, shape (3, H, W).
            reward: scalar (unused but required by API).

        Returns:
            np.ndarray([v, h, r], dtype=np.int32) with values in {0, 1, 2}.
        """
        try:
            obs = self.preprocessor.process(observation)

            visual = obs["visual"]
            vector = obs["vector"]

            obs_batch = {
                "visual": np.expand_dims(visual, axis=0),
                "vector": np.expand_dims(vector, axis=0),
            }

            action, _ = self.model.predict(obs_batch, deterministic=True)
            action = action.astype(np.int32).flatten()[:3]
            action = np.clip(action, 0, 2)

            # Mirror horizontal action back for left-side play
            if self.play_as_left:
                h = action[1]
                action[1] = 0 if h == 0 else 3 - h  # 0->0, 1->2, 2->1

            self.current_action = action
            return action

        except Exception as e:
            print(f"[TeamX] Error: {e}")
            return self.current_action.copy()
