# Phase 1: Wire Configs Through the Stack

## Goal

Make `env_kwargs` (`use_rgb`, `use_frame_diff`, `use_heatmap`, `frame_stack`) and
`reward_kwargs` (`aggression_bonus`, `hit_bonus`) actually flow from
experiment configs → environment → reward computation.

## Files Changed

| File | Changes | Lines |
|------|---------|-------|
| `env/wrapper.py` | Add `use_rgb`, `use_frame_diff`, `use_heatmap` params | +30 |
| `agent/wrapper_vec.py` | Thread `**env_kwargs` through factories | +6 |
| `agent/reward_shaping.py` | Add `aggression_bonus`, `hit_bonus` params | +12 |
| `run_experiment.py` | Wire `env_kwargs` + `reward_kwargs` into training | +15 |

---

## 1. `env/wrapper.py` — EnhancedUnityWrapper

### 1a. `__init__` changes

Add params after `use_ball_tracking`:

```python
def __init__(self, unity_env, frame_stack=4, use_ball_tracking=True,
             use_rgb=False, use_frame_diff=False, use_heatmap=False):
    self.env = unity_env
    self.agent = self.env.possible_agents[1]
    self.agent_other = self.env.possible_agents[0]
    self.agent_obs = self.env.possible_agents[0]
    self.frame_stack = frame_stack
    self.use_rgb = use_rgb
    self.use_frame_diff = use_frame_diff
    self.use_heatmap = use_heatmap

    base_obs = self.env.observation_spaces[self.agent_obs][0]
    c, h, w = base_obs.shape

    if use_rgb:
        obs_shape = (frame_stack * 3, h, w)
    else:
        obs_shape = (frame_stack, h, w)

    self.observation_space = spaces.Box(
        low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
    )
    ...
    self.frames = deque(maxlen=frame_stack)
    self.prev_raw_frame = None  # for frame_diff
```

### 1b. `_preprocess` changes

```python
def _preprocess(self, obs_tensor):
    if obs_tensor.ndim == 3 and obs_tensor.shape[0] in (3, 4):
        obs_tensor = obs_tensor.transpose(1, 2, 0)

    if self.use_rgb:
        obs_tensor = cv2.resize(obs_tensor, (168, 84), interpolation=cv2.INTER_AREA)
        obs_tensor = obs_tensor.astype(np.float32) / 255.0
        obs_tensor = obs_tensor.transpose(2, 0, 1)
        return obs_tensor

    gray = cv2.cvtColor(obs_tensor, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (168, 84), interpolation=cv2.INTER_AREA)
    gray = gray.astype(np.float32) / 255.0
    return np.expand_dims(gray, axis=0)


def _preprocess_frame_diff(self, obs_tensor):
    if obs_tensor.ndim == 3 and obs_tensor.shape[0] in (3, 4):
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
```

### 1c. `reset` and `step` changes

- In `reset`: if `use_frame_diff`, call `_preprocess_frame_diff` for init
- In `step`: if `use_frame_diff`, call `_preprocess_frame_diff` and stack diffs
- If `use_rgb`: stack 3-channel frames (`channels = frame_stack * 3`)
- If `use_heatmap`: after grayscale preprocess, get `ball_pos` from info,
  render 2D Gaussian overlay, then stack

---

## 2. `agent/wrapper_vec.py` — Thread kwargs

### 2a. Signatures

```python
def make_unity_env_fn(rank, build_path, frame_stack=4, use_ball_tracking=True,
                      base_port=5005, serve_mode=212, vec_env_cls=DummyVecEnv,
                      use_rgb=False, use_frame_diff=False, use_heatmap=False):
    def _thunk():
        ...
        env = EnhancedUnityWrapper(
            parallel,
            frame_stack=frame_stack,
            use_ball_tracking=use_ball_tracking,
            use_rgb=use_rgb,
            use_frame_diff=use_frame_diff,
            use_heatmap=use_heatmap,
        )
        return env
    return _thunk


def create_vec_env(build_path, num_envs=8, frame_stack=4,
                   use_ball_tracking=True, base_port=5005,
                   use_rgb=False, use_frame_diff=False, use_heatmap=False):
    env_fns = [
        make_unity_env_fn(
            rank=i, build_path=build_path,
            frame_stack=frame_stack,
            use_ball_tracking=use_ball_tracking,
            base_port=base_port,
            use_rgb=use_rgb,
            use_frame_diff=use_frame_diff,
            use_heatmap=use_heatmap,
        )
        for i in range(num_envs)
    ]
    return DummyVecEnv(env_fns)
```

---

## 3. `agent/reward_shaping.py` — Params

```python
def shaped_reward(sparse_reward, ball_pos, training_progress,
                  step=0, hit_detector=None,
                  paddle_x=0.35, paddle_y=0.5,
                  aggression_bonus=0.0, hit_bonus=0.5):
    dense_weight = annealing_weight(training_progress, end_phase=0.6)
    reward = sparse_reward
    if dense_weight > 0:
        reward += dense_weight * dense_distance_reward(ball_pos, paddle_x, paddle_y)
        if hit_detector is not None and hit_detector.update(ball_pos, step):
            reward += dense_weight * hit_bonus
    if aggression_bonus > 0 and ball_pos[0] > 0.5:
        reward += aggression_bonus
    return reward
```

---

## 4. `run_experiment.py` — Wire configs

```python
def run_experiment(config_name, total_steps=10_000, num_envs=2):
    config = resolve_config(config_name)
    env_kw = config.get("env_kwargs", {})
    reward_kw = config.get("reward_kwargs", {})

    vec_env = create_vec_env(
        build_path=BUILD_PATH,
        num_envs=num_envs,
        frame_stack=env_kw.get("frame_stack", 4),
        use_ball_tracking=True,
        use_rgb=env_kw.get("use_rgb", False),
        use_frame_diff=env_kw.get("use_frame_diff", False),
        use_heatmap=env_kw.get("use_heatmap", False),
    )
    ...
```

Add reward shaping inside `EnhancedUnityWrapper.step()`:

```python
# In EnhancedUnityWrapper.__init__:
self.reward_config = {
    "aggression_bonus": 0.0,
    "hit_bonus": 0.5,
}
self.hit_detector = HitDetector()

# In step():
agent_reward = rewards[self.agent] - rewards[self.agent_other]
if self.ball_tracker is not None:
    ball_pos = info.get("ball_pos", (-1.0, -1.0))
    agent_reward = shaped_reward(
        agent_reward, ball_pos, training_progress=1.0,
        step=self._step_count, hit_detector=self.hit_detector,
        **self.reward_config,
    )
```

Pass `reward_kw` from `create_vec_env` → `EnhancedUnityWrapper.reward_config`.

---

## Verification

```bash
python run_experiment.py --config baseline   --steps 2000 --envs 2
python run_experiment.py --config rgb_obs     --steps 2000 --envs 2
python run_experiment.py --config frame_diff  --steps 2000 --envs 2
```

- `baseline`: obs_shape = `(4, 84, 168)`, grayscale
- `rgb_obs`: obs_shape = `(12, 84, 168)`, 3-channel
- `frame_diff`: obs_shape = `(4, 84, 168)`, motion edges
- Heatmap: obs_shape = `(4, 84, 168)` with ball position overlay
