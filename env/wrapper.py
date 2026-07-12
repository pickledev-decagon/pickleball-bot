import numpy as np
import cv2
from collections import deque
from gymnasium import Env, spaces
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
from env.preprocessing import BallTracker, SmoothBallTracker, detect_paddles
from agent.reward_shaping import HitDetector, shaped_reward
from config import OBSERVATION_CONFIG


class EnhancedUnityWrapper(Env):
    def __init__(self, unity_env, frame_stack=16, use_ball_tracking=True,
                 use_rgb=False, use_frame_diff=False, use_heatmap=False,
                 heatmap_sigma=5, ##NEW
                 heatmap_weight=0.7, ##NEW
                 reward_config=None, opponent_policy=None, training_progress=0.0,
                 play_as_left=False, apply_reward_shaping=True,
                 mask_rotation=False,  # Fix 4: rotation masking for no-rotation training
                 max_rally_steps=500):  # stalemate termination (500 steps = 5s at 0.01s/step)
        
        self.heatmap_sigma = heatmap_sigma ##NEW
        self.heatmap_weight = heatmap_weight ##NEW
        self.apply_reward_shaping = apply_reward_shaping
        self.mask_rotation = mask_rotation  # Fix 4: force action[2]=0 when True
        self.max_rally_steps = max_rally_steps  # stalemate termination threshold
        self._steps_since_point = 0  # counter for stalemate detection

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

        # Dict observation space: visual (CNN) + vector (MLP)
        # vector = [ball_vx, ball_vy, paddle_x, paddle_y, ball_x, ball_y]
        self.observation_space = spaces.Dict({
            "visual": spaces.Box(
                low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
            ),
            "vector": spaces.Box(
                low=-1.0, high=1.0, shape=(6,), dtype=np.float32
            ),
        })
        raw_action_space = self.env.action_spaces[self.agent]
        import gymnasium as gym
        self.action_space = gym.spaces.MultiDiscrete(raw_action_space.nvec)

        self.frames = deque(maxlen=frame_stack)
        self.prev_raw_frame = None
        self._step_count = 0
        self.ball_tracker = BallTracker() if use_ball_tracking else None
        self.smooth_tracker = SmoothBallTracker() if use_ball_tracking else None
        self.hit_detector = HitDetector()
        self._last_ball_pos = (-1.0, -1.0)  # for opponent_policy (updated each step)
        self._last_opp_paddle_pos = (-1.0, -1.0)  # opponent's detected paddle pos (updated each step)
        self._last_ball_pos_for_vel = (-1.0, -1.0)  # for ball velocity computation
        self._last_vector_obs = np.zeros(6, dtype=np.float32)  # cached vector obs for opponent
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

    def _compute_vector_obs(self, info):
        """Compute 6-dim vector observation from detected ball/paddle positions.

        Returns: np.array([ball_vx, ball_vy, paddle_x, paddle_y, ball_x, ball_y],
                          dtype=float32)
        - ball_vx, ball_vy: velocity in normalized coords/step, scaled x10, clipped [-1,1]
          At 30fps, ball crossing court (1.0 units) in 1s = 30 steps -> vel ~0.033/step.
          Scaling x10 gives ~0.33, a usable range for the network.
        - paddle_x, paddle_y: our paddle position [0,1] or -1 if not detected
        - ball_x, ball_y: ball position [0,1] or -1 if not detected

        Must be called AFTER ball_pos mirroring (for left-side play) so the
        vector is in the same coordinate frame as the visual observation.
        """
        ball_pos = info.get("ball_pos", (-1.0, -1.0))
        paddle_pos = info.get("paddle_pos", (-1.0, -1.0))

        # Ball velocity from consecutive positions
        if self._last_ball_pos_for_vel[0] >= 0 and ball_pos[0] >= 0:
            ball_vx = np.clip(
                (ball_pos[0] - self._last_ball_pos_for_vel[0]) * 10, -1.0, 1.0
            )
            ball_vy = np.clip(
                (ball_pos[1] - self._last_ball_pos_for_vel[1]) * 10, -1.0, 1.0
            )
        else:
            ball_vx = 0.0
            ball_vy = 0.0
        self._last_ball_pos_for_vel = ball_pos

        return np.array(
            [ball_vx, ball_vy, paddle_pos[0], paddle_pos[1],
             ball_pos[0], ball_pos[1]],
            dtype=np.float32,
        )

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
            # Detect paddle positions (returns [(x,y), (x,y)] for 2 largest paddles)
            info["paddle_positions"] = detect_paddles(frame_uint8)
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
        self._steps_since_point = 0  # reset stalemate counter
        if self.hit_detector is not None:
            self.hit_detector.reset()

        # Compute vector observation (velocity = 0 on reset since no previous pos)
        self._last_ball_pos_for_vel = info.get("ball_pos", (-1.0, -1.0))
        vector_obs = self._compute_vector_obs(info)
        self._last_vector_obs = vector_obs

        visual_obs = np.concatenate(list(self.frames), axis=0)
        return {"visual": visual_obs, "vector": vector_obs}, info

    def step(self, action):
        # Mirror horizontal action if playing as left (model learned right-side coords)
        env_action = action.copy() if hasattr(action, 'copy') else np.array(action, copy=True)
        if self.play_as_left:
            h = env_action[1]  # index 1 = horizontal (order is [vertical, horizontal, rotation])
            env_action[1] = 0 if h == 0 else 3 - h  # 0->0 (none), 1->2 (right->left), 2->1 (left->right)

        # Fix 4: Mask rotation for no-rotation training (force action[2]=0)
        if self.mask_rotation:
            env_action[2] = 0

        actions = {self.agent: env_action}
        if self.opponent_policy is not None:
            # Opponent acts on the current stacked observation (obs_t).
            # Pass Dict obs (visual + vector) for SelfPlayOpponent compatibility.
            # Pass last-known ball_pos for ball-tracking opponents (backward compatible).
            current_visual = np.concatenate(list(self.frames), axis=0)
            current_obs = {"visual": current_visual, "vector": self._last_vector_obs}
            try:
                opp_action = self.opponent_policy(current_obs, ball_pos=self._last_ball_pos,
                                                  paddle_pos=self._last_opp_paddle_pos)
            except TypeError:
                try:
                    opp_action = self.opponent_policy(current_obs, ball_pos=self._last_ball_pos)
                except TypeError:
                    opp_action = self.opponent_policy(current_obs)
            # DEBUG: verify opponent action
            if self._step_count % 100 == 0:
                print(f"[DEBUG] Step {self._step_count}: our_action={env_action}  opp_action={opp_action}", flush=True)
            actions[self.agent_other] = opp_action

        obs_dict, rewards, terminations, infos = self.env.step(actions)

        raw_obs = obs_dict[self.agent_obs]['observation'][0]

        if self.use_frame_diff:
            obs = self._preprocess_frame_diff(raw_obs)
        else:
            obs = self._preprocess(raw_obs)

        info = self._build_info(raw_obs)

        # Update last-known ball_pos for next step's opponent_policy call.
        if "ball_pos" in info:
            self._last_ball_pos = info["ball_pos"]

        # Mirror ball_pos x-coordinate when playing left side so it matches
        # the already-flipped observation coordinate space.
        if self.play_as_left and "ball_pos" in info:
            bx, by = info["ball_pos"]
            info["ball_pos"] = (1.0 - bx, by) if bx >= 0 else (bx, by)

        # Select our paddle from detected positions and mirror if playing left.
        # Camera is always left-side (PAgent1): left paddle (smaller x) = PAgent1,
        # right paddle (larger x) = PAgent2.
        info["paddle_pos"] = (-1.0, -1.0)
        if "paddle_positions" in info:
            paddles = info["paddle_positions"]
            valid = [p for p in paddles if p[0] >= 0]
            if len(valid) >= 1:
                if self.play_as_left:
                    # Our paddle = PAgent1 = left paddle (smaller x), mirror x
                    ours = min(valid, key=lambda p: p[0])
                    px, py = ours
                    info["paddle_pos"] = (1.0 - px, py) if px >= 0 else (-1.0, -1.0)
                else:
                    # Our paddle = PAgent2 = right paddle (larger x), no mirror
                    ours = max(valid, key=lambda p: p[0])
                    info["paddle_pos"] = ours

        # Extract OPPONENT's paddle position (in camera space, no mirror)
        # for passing to opponent_policy on next step.
        # Opponent paddle = the OTHER paddle from ours.
        if "paddle_positions" in info:
            paddles = info["paddle_positions"]
            valid = [p for p in paddles if p[0] >= 0]
            if len(valid) >= 2:
                if self.play_as_left:
                    # Our paddle = left (smaller x), opponent = right (larger x)
                    self._last_opp_paddle_pos = max(valid, key=lambda p: p[0])
                else:
                    # Our paddle = right (larger x), opponent = left (smaller x)
                    self._last_opp_paddle_pos = min(valid, key=lambda p: p[0])
            elif len(valid) == 1:
                # Only one paddle detected — it's ours (already selected above)
                # Can't determine opponent position, keep previous estimate
                pass

        if self.use_heatmap:
            ball_pos = info.get("ball_pos", (-1.0, -1.0))
            obs = self._render_heatmap(obs, ball_pos)

        self.frames.append(obs)

        # Compute vector observation (after ball_pos mirroring for left-side play)
        vector_obs = self._compute_vector_obs(info)
        self._last_vector_obs = vector_obs

        agent_reward = rewards[self.agent] - rewards[self.agent_other]
        info["raw_reward"] = float(agent_reward)  # for eval rally segmentation
        if self.apply_reward_shaping and self.ball_tracker is not None:
            ball_pos = info.get("ball_pos", (-1.0, -1.0))
            paddle_pos = info.get("paddle_pos", (-1.0, -1.0))
            # Use detected paddle position if available, else fall back to config
            if paddle_pos[0] >= 0:
                reward_kwargs = {**self.reward_config, "paddle_x": paddle_pos[0],
                                 "paddle_pos_y": paddle_pos[1]}
            else:
                reward_kwargs = {**self.reward_config, "paddle_pos_y": None}
            agent_reward = shaped_reward(
                agent_reward, ball_pos, training_progress=self.training_progress,
                step=self._step_count, hit_detector=self.hit_detector,
                rotation_action=int(action[2]),
                **reward_kwargs,
            )
        self._step_count += 1
        terminated = terminations[self.agent]

        # Stalemate detection: terminate if no point scored for max_rally_steps
        if abs(float(agent_reward)) > 0.5:
            self._steps_since_point = 0  # point scored, reset counter
        else:
            self._steps_since_point += 1
        if self.max_rally_steps > 0 and self._steps_since_point >= self.max_rally_steps:
            terminated = True
            info["stalemate"] = True

        visual_obs = np.concatenate(list(self.frames), axis=0)
        return {"visual": visual_obs, "vector": vector_obs}, agent_reward, terminated, False, info

    def close(self):
        self.env.close()
