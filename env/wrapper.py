import numpy as np
import cv2
from collections import deque
from gymnasium import Env, spaces
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
from env.preprocessing import BallTracker, SmoothBallTracker
from agent.reward_shaping import HitDetector, shaped_reward
from config import OBSERVATION_CONFIG


class EnhancedUnityWrapper(Env):
    def __init__(self, unity_env, frame_stack=16, use_ball_tracking=True,
                 use_rgb=False, use_frame_diff=False, use_heatmap=False,
                 heatmap_sigma=5, ##NEW
                 heatmap_weight=0.7, ##NEW
                 reward_config=None, opponent_policy=None, training_progress=0.0,
                 play_as_left=False, apply_reward_shaping=True):
        
        self.heatmap_sigma = heatmap_sigma ##NEW
        self.heatmap_weight = heatmap_weight ##NEW
        self.apply_reward_shaping = apply_reward_shaping

        self.env = unity_env
        self.play_as_left = play_as_left

        if play_as_left:
            self.agent = self.env.possible_agents[0]
            self.agent_other = self.env.possible_agents[1]
        else:
            self.agent = self.env.possible_agents[1]
            self.agent_other = self.env.possible_agents[0]

        # Determine which agent provides the camera observation.
        # If the acting agent has a visual (≥2D) observation, use it per-agent.
        # Otherwise fall back to the other agent's camera (current build only
        # gives PAgent1 a camera; PAgent2 only has a vector observation).
        _obs_spaces = self.env.observation_spaces

        def _visual_space(sp):
            """Extract the 3D image Box from a space (Tuple or Box)."""
            # Tuple observation space, e.g. Tuple(Box(3,84,168), Box(1,))
            if hasattr(sp, "spaces"):
                for sub_sp in sp.spaces:
                    shape = getattr(sub_sp, "shape", ())
                    if len(shape) == 3:
                        return sub_sp
                return None

            # Direct Box observation space, e.g. Box(3,84,168) or Box(1,)
            shape = getattr(sp, "shape", ())
            if len(shape) == 3:
                return sp
            return None

        def _has_visual(aid):
            return _visual_space(_obs_spaces[aid]) is not None

        if _has_visual(self.agent):
            self.agent_obs = self.agent
        elif _has_visual(self.agent_other):
            self.agent_obs = self.agent_other
        else:
            raise ValueError(
                "No visual observation space found for either agent."
            )

        self.frame_stack = frame_stack
        self.use_rgb = use_rgb
        self.use_frame_diff = use_frame_diff
        self.use_heatmap = use_heatmap
        self.opponent_policy = opponent_policy
        self.training_progress = float(training_progress)

        base_obs = _visual_space(_obs_spaces[self.agent_obs])
        if base_obs is None:
            raise ValueError(
                "Selected observation source has no visual observation space."
            )
        c, h, w = base_obs.shape
        self._needs_transpose = (c == 3)

        if use_rgb:
            obs_shape = (frame_stack * 3, h, w)
        else:
            obs_shape = (frame_stack, h, w)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )
        raw_action_space = self.env.action_spaces[self.agent]
        import gymnasium as gym
        self.action_space = gym.spaces.MultiDiscrete(raw_action_space.nvec)

        self.frames = deque(maxlen=frame_stack)
        self.prev_raw_frame = None
        self._step_count = 0
        self.ball_tracker = BallTracker() if use_ball_tracking else None
        self.smooth_tracker = SmoothBallTracker() if use_ball_tracking else None
        self.hit_detector = HitDetector()
        _default_config = {
            "aggression_bonus": 0.0,
            "hit_bonus": 0.5,
            "paddle_x": OBSERVATION_CONFIG.effective_paddle_x,
        }
        self.reward_config = {**_default_config, **(reward_config or {})}

    def set_opponent_policy(self, policy_fn):
        self.opponent_policy = policy_fn

    def set_training_progress(self, progress):
        self.training_progress = float(progress)

    def _preprocess(self, obs_tensor):
        if self._needs_transpose:
            obs_tensor = obs_tensor.transpose(1, 2, 0)

        # If playing left side, horizontally flip so it looks like right side
        if self.play_as_left:
            obs_tensor = cv2.flip(obs_tensor, 1)

        # NOTE: Unity already returns float32 [0,1] — no /255.0 needed.
        if self.use_rgb:
            obs_tensor = cv2.resize(obs_tensor, (168, 84), interpolation=cv2.INTER_AREA)
            obs_tensor = obs_tensor.astype(np.float32)
            return obs_tensor.transpose(2, 0, 1)

        gray = cv2.cvtColor(obs_tensor, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (168, 84), interpolation=cv2.INTER_AREA)
        gray = gray.astype(np.float32)
        return np.expand_dims(gray, axis=0)

    def _preprocess_frame_diff(self, obs_tensor):
        if self._needs_transpose:
            obs_tensor = obs_tensor.transpose(1, 2, 0)
        curr_gray = cv2.cvtColor(obs_tensor, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.resize(curr_gray, (168, 84), interpolation=cv2.INTER_AREA)

        if self.prev_raw_frame is None:
            self.prev_raw_frame = curr_gray
            return np.zeros((1, 84, 168), dtype=np.float32)

        diff = cv2.absdiff(curr_gray, self.prev_raw_frame)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        self.prev_raw_frame = curr_gray
        return np.expand_dims(thresh.astype(np.float32) / 255.0, axis=0)

    # ⚠️ SYNC WARNING: Any change to this heatmap rendering logic MUST be
    # replicated in agent/competition_export.py:_render_heatmap().
    # The training wrapper and competition preprocessor must produce
    # identical preprocessing for the trained model to work correctly.
    def _render_heatmap(self, obs_tensor, ball_pos):
        if ball_pos[0] < 0:
            return obs_tensor
        h, w = obs_tensor.shape[1], obs_tensor.shape[2]
        gx = int(np.clip(ball_pos[0] * w, 0, w - 1))
        gy = int(np.clip(ball_pos[1] * h, 0, h - 1))
        gaussian = np.zeros((h, w), dtype=np.float32)

        ## Current — auto calculated, results in large sigma
        #sigma = max(1, min(h, w) // 16)
        ## With h=84, w=168: min(84,168)//16 = 84//16 = 5
        ## kernel_size = 6*5+1 = 31 pixels wide — very large on an 84px tall frame
        
        # Try this first — much tighter blob
        #sigma = 2
        #sigma = 3
        #sigma = 4
        sigma = self.heatmap_sigma ##NEW

        kernel_size = 6 * sigma + 1
        kx = cv2.getGaussianKernel(kernel_size, sigma)
        ky = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = kx @ ky.T
        kernel = kernel / kernel.max()  # Normalize peak to 1.0 so heatmap_weight controls blend
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
        #return np.clip(obs_tensor + heatmap * 0.5, 0.0, 1.0)
        ## NEW
        return np.clip(obs_tensor + heatmap * self.heatmap_weight, 0.0, 1.0) # see whether weight can improve or not

    def _build_info(self, obs_tensor):
        info = {}
        if self.ball_tracker is not None:
            frame = obs_tensor
            if self._needs_transpose:
                frame = frame.transpose(1, 2, 0)
            frame_uint8 = (frame * 255).astype(np.uint8)
            detected = self.ball_tracker.detect(frame_uint8)
            # Apply temporal smoothing to handle brief occlusions
            if self.smooth_tracker is not None:
                info["ball_pos"] = self.smooth_tracker.update(detected)
            else:
                info["ball_pos"] = detected
        return info

    def reset(self, *, seed=None, options=None):
        obs_dict = self.env.reset()
        raw_obs = obs_dict[self.agent_obs]['observation'][0]

        if self.use_frame_diff:
            obs = self._preprocess_frame_diff(raw_obs)
        else:
            obs = self._preprocess(raw_obs)

        info = self._build_info(raw_obs)

        # Mirror ball_pos x-coordinate when playing left side so it matches
        # the already-flipped observation coordinate space.
        if self.play_as_left and "ball_pos" in info:
            bx, by = info["ball_pos"]
            info["ball_pos"] = (1.0 - bx, by) if bx >= 0 else (bx, by)

        if self.use_heatmap:
            ball_pos = info.get("ball_pos", (-1.0, -1.0))
            obs = self._render_heatmap(obs, ball_pos)

        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(obs)

        self.prev_raw_frame = None
        self._step_count = 0
        if self.hit_detector is not None:
            self.hit_detector.reset()
        return np.concatenate(list(self.frames), axis=0), info

    def step(self, action):
        # Mirror horizontal action if playing as left (model learned right-side coords)
        env_action = action.copy() if hasattr(action, 'copy') else np.array(action, copy=True)
        if self.play_as_left:
            h = env_action[1]
            env_action[1] = 2 - h  # 0<->2, 1 stays 1

        actions = {self.agent: env_action}
        if self.opponent_policy is not None:
            # Opponent acts on the current stacked observation (obs_t)
            current_obs = np.concatenate(list(self.frames), axis=0)
            opp_action = self.opponent_policy(current_obs)
            actions[self.agent_other] = opp_action

        obs_dict, rewards, terminations, infos = self.env.step(actions)

        raw_obs = obs_dict[self.agent_obs]['observation'][0]

        if self.use_frame_diff:
            obs = self._preprocess_frame_diff(raw_obs)
        else:
            obs = self._preprocess(raw_obs)

        info = self._build_info(raw_obs)

        # Mirror ball_pos x-coordinate when playing left side so it matches
        # the already-flipped observation coordinate space.
        if self.play_as_left and "ball_pos" in info:
            bx, by = info["ball_pos"]
            info["ball_pos"] = (1.0 - bx, by) if bx >= 0 else (bx, by)

        if self.use_heatmap:
            ball_pos = info.get("ball_pos", (-1.0, -1.0))
            obs = self._render_heatmap(obs, ball_pos)

        self.frames.append(obs)

        agent_reward = rewards[self.agent] - rewards[self.agent_other]
        if self.apply_reward_shaping and self.ball_tracker is not None:
            ball_pos = info.get("ball_pos", (-1.0, -1.0))
            agent_reward = shaped_reward(
                agent_reward, ball_pos, training_progress=self.training_progress,
                step=self._step_count, hit_detector=self.hit_detector,
                rotation_action=int(action[2]),
                **self.reward_config,
            )
        self._step_count += 1
        terminated = terminations[self.agent]

        return np.concatenate(list(self.frames), axis=0), agent_reward, terminated, False, info

    def close(self):
        self.env.close()
