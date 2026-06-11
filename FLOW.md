# Project Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  UNITY ENVIRONMENTS ×8             [analogy: 8 pickleball courts]           │
│                                                                             │
│  ┌────────────────┐ ┌────────────────┐               ┌────────────────┐    │
│  │  worker_id=0   │ │  worker_id=1   │    ...        │  worker_id=7   │    │
│  │  port=5005     │ │  port=5006     │               │  port=5012     │    │
│  │  (3,84,168)    │ │  (3,84,168)    │               │  (3,84,168)    │    │
│  └───────┬────────┘ └───────┬────────┘               └───────┬────────┘    │
│          │                  │                                │              │
│          └──────────────────┴────────────────────────────────┘              │
│                              │                                             │
│                     gRPC (20-29ms/step)                                     │
└─────────────────────────────┼───────────────────────────────────────────────┘
                              │
                              ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  EnhancedUnityWrapper           [analogy: your eyes + memory]           │
  │                                                                         │
  │  [1] _preprocess(): transpose → resize(84,168) → RGB→GRAY → normalize  │
  │      obs: (3,84,168) ──────────────────────────────────► (1,84,168)     │
  │                                                                         │
  │  [2] Stack last 4 grayscale frames in deque(maxlen=4)                  │
  │      stacked_obs: (1,84,168)×4 ──────────────────────► (4,84,168)      │
  │                                                                         │
  │  [3] BallTracker                    [analogy: spotter yells ball pos]   │
  │      RGB→HSV → inRange(H=20-35,S=80-255,V=80-255) → erode+dilate       │
  │      → findContours → largest → moments → (ball_x, ball_y)             │
  │      Result: info["ball_pos"] = (x, y) or (-1,-1)                      │
  │                                                                         │
  │  [4] Reward: agent_reward = r_self - r_opponent                        │
  └──────────────────────────────────┬──────────────────────────────────────┘
                                     │
                                     ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  DummyVecEnv ×8                  [analogy: 8 parallel practice courts]  │
  │  In-process sequential stepping. No IPC overhead.                       │
  │  Output: obs=(8,4,84,168), reward=(8,), done=(8,)                       │
  └──────────────────────────────────┬──────────────────────────────────────┘
                                     │
                                     ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Reward Shaping                  [analogy: coach's realtime feedback]    │
  │                                                                         │
  │  Annealing weight w_dense:                                              │
  │    progress 0.0 ──────► 0.6 ─────────────► 1.0                         │
  │    w_dense   1.0  linear decay  0.0  (no dense reward after 60%)        │
  │                                                                         │
  │  r_total = r_sparse + w_dense * (                                       │
  │      -0.01 * distance(ball_pos, paddle)   [proximity reward]            │
  │      + 0.5  * hit_detected               [paddle contact bonus]         │
  │  )                                                                      │
  │                                                                         │
  │  HitDetector: velocity_change = |Δx₂ - Δx₁| > 0.02 → hit               │
  └──────────────────────────────────┬──────────────────────────────────────┘
                                     │
                                     ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  CurriculumCallback              [analogy: progressive training plan]    │
  │                                                                         │
  │  ┌──────────────┐    ┌──────────────┐    ┌─────────────────────┐       │
  │  │  STAGE 1     │    │  STAGE 2     │    │  STAGE 3            │       │
  │  │  0-30%       │───►│  30-60%      │───►│  60-100%            │       │
  │  │  Opponent:   │    │  Opponent:   │    │  Opponent: random   │       │
  │  │  no-op [0]   │    │  square pat. │    │  from buffer        │       │
  │  │  Dense: on   │    │  Dense: fade │    │  Dense: off         │       │
  │  └──────────────┘    └──────────────┘    └──────────┬──────────┘       │
  │  [practice vs wall]  [practice vs robot] [practice vs past selves]      │
  │                                                      │                  │
  │                                                      ▼                  │
  │                                          ┌──────────────────────┐       │
  │                                          │ OpponentBuffer       │       │
  │                                          │ [sparring partner    │       │
  │                                          │  pool: 5 checkpoints]│       │
  │                                          │ random sample        │       │
  │                                          │ worst evicted        │       │
  │                                          └──────────────────────┘       │
  └──────────────────────────────────┬──────────────────────────────────────┘
                                     │
                                     ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  SB3 PPO (CnnPolicy)             [analogy: the athlete's brain]         │
  │                                                                         │
  │  ┌─ CNN (visual cortex) ──────────────────────────────────────────┐    │
  │  │  (4,84,168) → Conv2D(32,8×8,stride4) → ReLU → Conv2D(64,4×4)  │    │
  │  │  → ReLU → Conv2D(64,3×3,stride1) → ReLU → FC(512) → ReLU      │    │
  │  └────────────────────────────────────────────────────────────────┘    │
  │                      │                                                  │
  │              ┌───────┴────────┐                                         │
  │      actor   ▼                ▼   critic                                │
  │  ┌─────────────────┐  ┌──────────────────┐                              │
  │  │ decision reflex  │  │ internal judge   │                              │
  │  │ FC(3×3)         │  │ FC(1) → V(s)     │                              │
  │  │ softmax → action│  │ "how good is     │                              │
  │  │ "move here!"    │  │  this state?"    │                              │
  │  └─────────────────┘  └──────────────────┘                              │
  │                                                                         │
  │  Hyperparameters: lr=2.5e-4, gamma=0.99, batch=128, ent_coef=0.01       │
  │  Updates: every n_steps=2048 total steps (256 per env × 8 envs)         │
  │  Device: cpu (Unity 20ms per step = bottleneck, not inference)          │
  └──────────────────────────────────┬──────────────────────────────────────┘
                                     │
                                     ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Checkpoint save/load            [analogy: muscle memory journal]        │
  │  model.save() → .zip file       │  model.load() → resume training       │
  │  OpponentBuffer stores past versions as sparring partners               │
  └──────────────────────────────────┬──────────────────────────────────────┘
                                     │
                                     ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Experiment Runner               [analogy: training lab]                │
  │                                                                         │
  │  python run_experiment.py --config <name> --steps N --envs M            │
  │                                                                         │
  │  Configs in agent/experiment_configs.py (13 presets):                   │
  │  baseline / entropy_0.0 / entropy_0.05 / rgb_obs / frame_diff /        │
  │  heatmap_obs / aggression_bonus / no_curriculum / large_batch /         │
  │  small_batch / hit_bonus_high / lr_1e-4 / stack_2                      │
  │                                                                         │
  │  Output: results/<config>_<timestamp>.json                              │
  └──────────────────────────────────┬──────────────────────────────────────┘
                                     │
                                     ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Competition Export              [analogy: game day — compete with only │
  │                                   your reflexes, no training gear]       │
  │  Evaluate: python agent/evaluate.py --checkpoint model.zip              │
  │                                                                         │
  │  Export: competition/teamX.py / teamY.py                                │
  │  - Loads saved PPO checkpoint                                           │
  │  - model.predict(obs, deterministic=True) per frame                     │
  │  - Must respond in < 0.01s                                              │
  │  - Only numpy, opencv, torch allowed                                    │
  └─────────────────────────────────────────────────────────────────────────┘
```

## Data Shapes at Each Step

| Stage | Shape | Type | Example |
|---|---|---|---|
| Unity raw obs | `(3, 84, 168)` | float32 `[0,1]` | RGB camera frame |
| After grayscale | `(1, 84, 168)` | float32 `[0,1]` | Single luminance channel |
| Frame stack (×4) | `(4, 84, 168)` | float32 `[0,1]` | 4 consecutive frames |
| VecEnv batch | `(8, 4, 84, 168)` | float32 `[0,1]` | 8 parallel envs |
| CNN feature | `(512,)` | float32 | After conv layers + FC |
| Action logits | `(3, 3)` | float32 | 3 branches × 3 values |
| Sampled action | `(3,)` | int32 `{0,1,2}` | [vertical, horizontal, rotate] |
| Reward | `(1,)` | float32 | Scalar signal |
| Ball coords | `(2,)` | float32 `[0,1]` | `(ball_x, ball_y)` or `(-1,-1)` |

## Quick Reference

```
Unity ──► EnvWrapper ──► VecEnv(×8) ──► RewardShape ──► Curriculum ──► PPO
(court)    (eyes+mem)    (8 courts)      (coach)         (training      (brain)
                                                          plan)
```

**Training lifecycle:**
1. 8 Unity envs each produce `(3,84,168)` obs
2. Wrapper converts to grayscale, stacks 4 frames → `(4,84,168)`, detects ball
3. DummyVecEnv batches everything → `(8,4,84,168)`
4. CurriculumCallback selects opponent (no-op → square → buffer)
5. RewardShaping applies w_dense * (proximity + hit) + sparse
6. PPO collects 2048 total steps, runs 4 epochs of gradient updates
7. Checkpoints saved every 10% of training → OpponentBuffer
8. Best checkpoint exported to competition/ for submission

---

## Concise Summary

| Aspect | Summary |
|---|---|
| **Goal** | Train a pickleball RL agent that wins a course competition |
| **Input** | `84×168` RGB camera frame → grayscale → 4-frame stack → `(4,84,168)` |
| **Output** | 3 discrete actions: vertical (up/down/idle), horizontal (left/right/idle), rotate (cw/idle/ccw) |
| **Algo** | PPO via SB3 with CnnPolicy, `device="cpu"`, `gamma=0.99`, `ent_coef=0.01` |
| **Speed** | Unity step = 20-29ms (bottleneck), model inference = 1-2ms, ~30-50 FPS on M1 (2 envs) |
| **Parallel** | 8× DummyVecEnv, each with its own Unity instance (unique worker_id, port) |
| **Ball** | HSV yellow detection `H=20-35 S=80-255 V=80-255`, 98-100% accuracy, coords in `info["ball_pos"]` |
| **Reward** | Sparse `+1/-1` + dense proximity `-0.01*dist` + hit bonus `+0.5`, annealed to zero by 60% of training |
| **Curriculum** | Stage 1 (no-op opponent, dense reward) → Stage 2 (square opponent, reward fades) → Stage 3 (self-play buffer, sparse only) |
| **Self-play** | OpponentBuffer of 5 checkpoints, randomly sampled, worst evicted |
| **Train time** | ~1-2 hours for 150k steps × 8 envs = 1.2M frames |
| **Experiment** | `run_experiment.py` — 13 named configs, JSON output to `results/` |
| **Export** | `model.predict(obs, deterministic=True)` in `competition/teamX.py`, < 0.01s per step |

---

## Pipeline Analogy

| Component | Code File | Analogy |
|---|---|---|
| **Unity Environment** | `Unity app` | **8 pickleball courts.** Ball physics, paddles, scoring — the real game. |
| **EnhancedUnityWrapper** | `env/wrapper.py` | **Your eyes + short-term memory.** Sharpens the image, converts to simple grayscale, remembers last 4 frames so you see motion. |
| **BallTracker** | `env/preprocessing.py` | **A spotter** who tracks the ball and yells its position, so you don't search the whole frame. |
| **DummyVecEnv ×8** | `agent/wrapper_vec.py` | **8 parallel practice courts.** 8× more experience per hour than a single court. |
| **Reward Shaping** | `agent/reward_shaping.py` | **Coach's realtime feedback.** While the game only says "you won/lost" after 200 shots, the coach gives small feedback each step: "good, you moved toward the ball." As you improve, the coach fades out. |
| **CurriculumCallback** | `agent/train_selfplay_v2.py` | **Progressive training plan.** Stage 1: practice against a wall. Stage 2: practice against a predictable robot. Stage 3: practice against your past selves. |
| **OpponentBuffer** | `agent/buffer.py` | **Sparring partner pool.** 5 past versions of yourself. Coach randomly picks one each game so you don't overfit to one play style. |
| **SB3 PPO** | (library) | **The athlete yourself.** After each practice session, you review what worked and strengthen the mental pathways that led to good outcomes. |
| **CnnPolicy** | (inside PPO) | **Your visual cortex.** Specialized neurons that recognize court lines, ball, paddles from raw pixels. |
| **Actor head** | (inside CnnPolicy) | **Your decision reflex.** "Move up, angle left" — the output sent to your muscles. |
| **Critic head** | (inside CnnPolicy) | **Your internal judge.** "I'm in a good position, this will go well" — estimates whether the current situation is promising. |
| **Checkpoint save/load** | `model.save()` / `.load()` | **Muscle memory journal.** Write down everything you learned. Reload it later — or your past self can spar with your current self. |
| **run_experiment.py** | `run_experiment.py` | **Training lab.** Tests different strategies: "what if I train with color vision? What if I explore more?" Logs and compares results. |
| **Competition export** | `competition/teamX.py` | **Game day.** Freeze your best version. Strip out all training gear. Compete with just reflexes — `model.predict()`, no learning, 0.01s to react. |
