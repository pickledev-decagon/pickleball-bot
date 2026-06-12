# dPickleBall Bot — Technical Guide

<!-- This is the single-source-of-truth for all technical decisions, architecture, and usage. -->

---

## 1. Core Architecture

```
                  action [3 discrete: Y, X, Rot]
                        │
  ┌─────────────────────▼─────────────────────┐
  │  SB3 PPO (CnnPolicy)                      │
  │  ┌──────────┐   ┌──────────────────────┐  │
  │  │ CNN      │   │ PPO Head:            │  │
  │  │ conv 32  │   │ actor logits → action│  │
  │  │ conv 64  │──►│ critic stream → V(s) │  │
  │  │ conv 64  │   └──────────────────────┘  │
  │  │ FC 512   │                             │
  │  └──────────┘                             │
  └─────────────────────▲─────────────────────┘
                        │
        obs (4,84,168)  │  reward (scalar)
        ┌───────────────┴──────┐
        │  EnhancedUnityWrapper │
        │  ┌─────────┐ ┌─────┐ │
        │  │ BallTracker │frame│ │
        │  │ (HSV→coords)│stack│ │
        │  └─────────┘ └─────┘ │
        └───────────────▲──────┘
                        │
        Unity env via gRPC
  ┌──────────────────────┴──────────────────────┐
  │  DummyVecEnv × 8                             │
  │  (worker_id=0..7, ports 5005..5012)          │
  └──────────────────────────────────────────────┘
```

### Observation Pipeline

Raw Unity observations arrive as `(3, 84, 168)` float32 in `[0, 1]`.

```python
# env/wrapper.py — EnhancedUnityWrapper._preprocess()
obs_tensor = obs_tensor.transpose(1, 2, 0)        # (84,168,3)
obs_tensor = cv2.resize(obs_tensor, (168, 84))     # ensure 84×168
obs_tensor = cv2.cvtColor(obs_tensor, cv2.COLOR_RGB2GRAY)  # (84,168) grayscale
```

4 consecutive grayscale frames are stacked → `(4, 84, 168)` observation. This gives the CNN motion perception (ball direction, speed) from static frames.

### BallTracker Algorithm

`env/preprocessing.py:BallTracker` detects the yellow ball in each frame:

1. **Convert** RGB → HSV (`cv2.COLOR_RGB2HSV`)
2. **Threshold** via `cv2.inRange()` with:
   - Hue: `[20, 35]` (yellow range)
   - Saturation: `[80, 255]`
   - Value: `[80, 255]`
3. **Clean** binary mask: `erode(1×)` then `dilate(2×)` with `3×3` kernel
4. **Find contours** with `RETR_EXTERNAL + CHAIN_APPROX_SIMPLE`
5. **Filter** by contour area: reject if area ≤ 5 or ≥ 200 pixels
6. **Centroid** via image moments on the largest valid contour
7. **Normalize** to `[0, 1]` by dividing by frame width/height

Detection rate: **98-100%** on live Unity frames. Returns `(x, y)` or `(-1, -1)` if ball is off-screen/undetected. Ball position is included in `info["ball_pos"]` for reward shaping.

---

## 2. Reward Shaping

The environment gives sparse rewards: `+1` when you score, `-1` when opponent scores. Between points, reward is `0`. This makes learning extremely slow — the agent plays hundreds of steps per point with zero feedback.

### Shaped Reward Formula

```python
# agent/reward_shaping.py
r_total = r_sparse + w_dense * (r_proximity + r_hit)
```

Where:

| Component | Formula | Purpose |
|---|---|---|
| `r_sparse` | `+1` (scored) / `-1` (conceded) | Win/loss signal — always available |
| `w_dense` | Annealing weight (see below) | Controls how much dense reward is added |
| `r_proximity` | `-0.01 * sqrt((x_ball - 0.35)² + (y_ball - 0.5)²)` | Encourages moving toward ball |
| `r_hit` | `+0.5` on detected paddle-ball contact | Reinforces successful hits |

Proximity reward is only nonzero when the ball is on our side of the court (`ball_x > 0`). `(0.35, 0.5)` approximates our paddle's home position.

### Annealing Schedule

```python
def annealing_weight(progress):
    if progress < 0.0:
        return 1.0      # first 0% (start)
    elif progress > 0.6:
        return 0.0      # after 60% — no dense reward
    else:
        return 1.0 - progress / 0.6   # linear decay
```

**Purpose:** Dense reward acts as training wheels — it guides exploration early, then fades to zero so the agent is evaluated purely on competitive win/loss. This prevents the agent from optimizing for proximity (dinking) instead of scoring.

### Hit Detection

`HitDetector` tracks the last 5 ball positions. A hit is detected when:

```python
# ball x-velocity changes by more than 0.02 between consecutive frames
abs((x₂ - x₁) - (x₁ - x₀)) > 0.02
```

Cooldown: 10 steps between hit detections to prevent double-counting a single hit.

---

## 3. Curriculum Stages

Training is divided into 3 stages, each building on the previous:

```
Progress    Stage  Opponent           Reward Structure
────────────────────────────────────────────────────────────
  0-30%      1     No-op (idle)      Dense only (w_dense=1.0)
 30-60%      2     Square motion     Dense decaying (w_dense: 1.0→0)
 60-100%     3     Self-play buffer  Sparse only (w_dense=0)
```

### Stage 1: No-Op Opponent (0-30%)

The opponent paddle does nothing — always outputs action `[0, 0, 0]` (no move, no spin). The agent learns basic paddle control: how to move toward the ball and make contact. Dense reward is fully active.

**Goal:** Learn that `moving toward ball → positive feedback`.

### Stage 2: Square-Motion Opponent (30-60%)

The opponent moves in a fixed square pattern: down, right, up, left, repeating every 120 steps. This introduces mild defensive pressure. The agent must learn to track a moving ball and time its hits.

Dense reward weight anneals from `1.0 → 0.0` during this stage, gradually shifting the objective from "get near the ball" to "win the point."

### Stage 3: Self-Play (60-100%)

The opponent is a past checkpoint from the OpponentBuffer, sampled randomly. Only sparse reward `+1/-1` is used. The agent must learn general strategies that work against diverse play styles.

**Key design choice:** Random sampling (not always the best opponent) prevents the agent from overfitting to a single strategy. This follows the OpenAI competitive self-play approach.

---

## 4. Self-Play Mechanics

### OpponentBuffer

`agent/buffer.py:OpponentBuffer` maintains a pool of past checkpoints:

| Parameter | Value | Rationale |
|---|---|---|
| `max_size` | 5 | Keeps training diverse without consuming disk space |
| Sampling | Random | Maximizes opponent diversity |
| Eviction | Worst win_rate | Keeps only the strongest opponents |
| Persistence | `manifest.json` | Survives restarts |

```python
buffer = OpponentBuffer(save_dir, max_size=5)
buffer.add("checkpoint_100k.zip", win_rate=0.62)
opponent = buffer.sample()  # random, weighted by win_rate
```

Checkpoints are added at the end of each training run. The buffer is also filled gradually during Stage 3 — as the agent trains, new checkpoints are dropped into the buffer.

### What the Opponent Does at Each Stage

| Stage | Opponent | Implementation |
|---|---|---|
| 1 | No-op | `np.array([0, 0, 0])` per step |
| 2 | Square motion | Predefined `(↓ → ↑ ←)` sequence, state-machine per parallel env |
| 3 | Self-play | `PPO.load(ckpt).predict(obs, deterministic=True)[0]` |

Stage 3's self-play policy is loaded lazily (only when a new checkpoint is sampled) and cached until resampling. Running a full PPO policy for the opponent costs ~1-2ms, negligible next to Unity's 20ms step time.

---

## 5. Parallel Environments

### VecEnv Design

```python
# agent/wrapper_vec.py
8 instances of UnityEnvironment with:
  - worker_id = 0..7 (unique per env)
  - base_port = 5005 → ports 5005..5012
  - DummyVecEnv (in-process, no IPC overhead)
```

**Why DummyVecEnv over SubprocVecEnv:**
- Unity environments use gRPC for communication (not shared memory)
- SubprocVecEnv spawns separate processes just to call `env.step()` — adds IPC overhead with no benefit
- DummyVecEnv runs envs sequentially in one process, which is fine since Unity step time (~20ms) dominates

### Throughput

| Setup | Steps/sec (2 envs on M1) | Steps/sec (8 envs on M5 Pro) |
|---|---|---|
| Theoretical max | 100 (50ms per env at 20Hz) | 400 |
| Actual measured | 30-50 | ~200-300 (estimated) |

Unity environment step takes **20-29ms** (includes physics + rendering + gRPC). Model inference is **1-2ms**. CPU vs GPU makes no meaningful difference — we run `device="cpu"`.

---

## 6. PPO Configuration

Hyperparameters chosen based on SB3 defaults (which match the 37 PPO Implementation Details paper):

| Parameter | Value | Why |
|---|---|---|
| `learning_rate` | 2.5e-4 | SB3 Atari default — proven stable |
| `n_steps` | `256 × num_envs` | 2048 total steps per update |
| `batch_size` | 128 | Good gradient stability |
| `n_epochs` | 4 | SB3 default |
| `gamma` | 0.99 | Long episodes (500-2000 steps) — need long horizon |
| `gae_lambda` | 0.95 | Standard GAE |
| `clip_range` | 0.2 | Standard PPO clip |
| `ent_coef` | 0.01 | Encourages exploration in sparse-reward setting |
| `vf_coef` | 0.5 | Standard value/policy loss balance |
| `max_grad_norm` | 0.5 | SB3 default |
| `device` | `cpu` | Unity rendering is bottleneck, not inference |

**Why high gamma (0.99):** Episodes are long (500-2000 steps to reach 21 points). A lower gamma would heavily discount distant future rewards, making the agent short-sighted.

**Why ent_coef 0.01:** Sparse reward environments need exploration. TOO much entropy (0.05) keeps the agent random; too little (0.0) risks premature convergence.

---

## 7. How to Read Training Output

### TensorBoard Metrics

Launch TensorBoard:
```bash
tensorboard --logdir logs/
```

Key metrics in the SCALARS tab:

| Metric | Healthy Range | What It Tells You |
|---|---|---|
| `train/entropy` | `-3.5` to `-2.5` | Policy randomness. Starts near action-space entropy (`-ln(1/3)≈-1.1` per branch ×3 = `-3.3`). Decreasing = policy is becoming more deterministic. |
| `train/policy_loss` | Noisy, ±0.02 | Objective being maximized. Should NOT smoothly increase — that means the policy isn't changing enough. |
| `train/value_loss` | Decreasing → low | Bellman error. Should decrease as the critic's value estimates improve. |
| `train/explained_variance` | Increasing → 0.3+ | How well the critic explains observed returns. 0.0 = no better than guessing, 1.0 = perfect. |
| `train/approx_kl` | `< 0.01` | How much the policy changed in this update. `6e-6` = tiny changes (stable). `> 0.02` = potentially destabilizing. |
| `rollout/ep_len_mean` | Varies | Average episode length. Decreasing = points being scored faster. |
| `rollout/ep_rew_mean` | Increasing | Average episode return. Should trend upward. |

### Console Log Breakdown

```
----------------------------------
| rollout/           |          |
|    ep_len_mean     | 1.1e+03 |
|    ep_rew_mean     | -1.76   |
| time/              |          |
|    total_timesteps | 100352  |
|    fps             | 31      |
----------------------------------
| train/             |          |
|    entropy         | -3.28   |
|    policy_loss     | 0.0081  |
|    value_loss      | 0.165   |
|    approx_kl       | 6.11e-6 |
|    clip_fraction   | 0.0639  |
|    explained_variance | 0.0113  |
----------------------------------
```

**Quick sanity check flow:**
1. `entropy` ≈ `-3.3` → good, agent is still exploring
2. `approx_kl` < `0.01` → updates are stable
3. `explained_variance` trending up → critic is learning
4. `fps` consistent → no bottleneck

### Diagnostic Rules

| Pattern | Diagnosis | Action |
|---|---|---|
| `entropy` crashes to `-0.5` | Policy collapsed. Stops exploring. | Lower `ent_coef` or increase clip_range |
| `approx_kl` > `0.02` | Policy changed too much in one update. Unstable. | Lower `lr`, increase `target_kl` |
| `value_loss` increasing | Critic diverging from true returns | Lower `lr` or `vf_coef` |
| `explained_variance` stuck at `0.0` | Critic not learning anything | Check reward signal is nonzero |
| `fps` drops halfway | Garbage collection or Unity slowdown | Reduce `num_envs` |
| `ep_rew_mean` stays flat | Agent not improving | Is there any reward signal? Check reward shaping |

---

## 8. How to Pick the Best Checkpoint

Checkpoints are saved every `total_steps // 10` steps during training (`~/opponent_buffer_v2/`).

**Best practice:**
1. Run evaluation: `python agent/evaluate.py --checkpoint <path>.zip --episodes 20`
2. Select the checkpoint with the highest win rate against a fixed opponent
3. The **final checkpoint** is NOT always the best — sometimes an intermediate checkpoint generalized better before overfitting

The OpponentBuffer automatically tracks win rates and sorts checkpoints:
```python
buffer.get_best()   # checkpoint with highest win_rate
buffer.get_worst()  # checkpoint with lowest win_rate
```

---

## 9. Experiment Runner

All experiments use `run_experiment.py` — a shared harness that loads named configs from `agent/experiment_configs.py`.

```bash
# List all available experiments
python run_experiment.py --list

# Run an experiment for 10k steps across 4 parallel envs
python run_experiment.py --config baseline --steps 10000 --envs 4

# Run with different hyperparameters
python run_experiment.py --config entropy_0.05 --steps 20000 --envs 8
```

Output is saved to `results/<config>_<timestamp>.json` with keys:
```json
{
  "experiment": "entropy_0.05",
  "description": "ent_coef=0.05 — maximum exploration",
  "timestamp": "20260513_163541",
  "total_steps": 10000,
  "num_envs": 4,
  "elapsed_seconds": 312.5,
  "steps_per_second": 32.0,
  "train/entropy": -2.85,
  "train/policy_loss": 0.0062,
  "train/value_loss": 0.142,
  "train/approx_kl": 7.3e-6
}
```

Export to RESULTS.md after each run for comparison.

---

## 10. Competition Export

The competition requires a Python file with a `policy(obs, reward)` function. Your file:

1. Loads the saved PPO checkpoint
2. Runs `model.predict(obs, deterministic=True)` per frame
3. Respects the 0.01s response time limit

**Template** (will be placed in `competition/`):

```python
import numpy as np
from stable_baselines3 import PPO

model = PPO.load("path/to/checkpoint.zip", device="cpu")

reward = 0
def policy(obs, r):
    global reward
    reward = r if r != 0 else reward
    action, _ = model.predict(obs, deterministic=True)
    return action
```

Key constraint: **no extra Python packages.** Only numpy, opencv, torch — all available in the competition environment. See [competition rules in OVERVIEW.md](OVERVIEW.md).

---

## 11. Research Justification

| Source | Key Finding | How We Apply |
|---|---|---|
| **37 PPO Implementation Details** (ICLR Blog Track, 2022) | SB3 matches reference PPO with all critical optimizations | Use SB3 defaults, don't fight the framework |
| **Competitive Self-Play** (OpenAI, 2017) | Opponent diversity prevents overfitting; dense reward annealing helps | OpponentBuffer with random sampling; reward schedule |
| **Implementation Matters in Deep Policy Gradients** (ICLR, 2020) | Code details matter more than algorithm choice | Focus on obs preprocessing, reward shaping, curriculum — not PPO variants |
| **Lecturer feedback** | CPU=GPU for this env; 8 envs sweet spot; curriculum helps | All three applied in v2 |

---

## 12. Experiment Catalog

Every approach we try, with hypothesis, risk, and validation metric.
Detailed per-member experiment logs are in `progress/member-*-area.md`.

### Metrics Interpretation Primer

| Metric | What It Measures | Good Sign | Bad Sign |
|--------|-----------------|-----------|----------|
| `entropy` | Policy randomness. Starts at `-3.3` (max entropy for `3×3` actions). | `-3.5` to `-2.5` (slowly decreasing) | `> -1.5` = collapsed; `< -3.5` = still random |
| `approx_kl` | How much the policy changed in this update | `< 0.01` (stable) | `> 0.02` (destabilizing) |
| `value_loss` | Critic prediction error | Decreasing | Increasing (critic diverging) |
| `explained_variance` | How well critic explains returns | Increasing toward `1.0` | Stuck at `0.0` (critic learning nothing) |
| `policy_loss` | Policy gradient magnitude | Noisy, non-zero | Constant zero (policy not updating) |
| `fps` | Training speed | Consistent | Dropping (bottleneck) |
| `ep_rew_mean` | Avg episode return | Trending up | Flat |
| `ep_len_mean` | Avg episode length | Decreasing (faster points) | Increasing (longer rallies) |

### Decision Tree

```
entropy crashed to -1.0 early?
├─ Yes → ent_coef too low, try entropy_0.05 or entropy_0.01 (baseline)
└─ No  → keep current or try entropy_0.0

approx_kl > 0.02?
├─ Yes → updates too large, lower lr → try lr_1e-4
└─ No  → can try higher lr or baseline

fps lower than expected?
├─ Yes → reduce obs complexity → try stack_2 or grayscale (baseline)
└─ No  → can try rgb_obs or more envs

ep_rew_mean not improving?
├─ Ball never hit? → check reward shaping → try hit_bonus_high
├─ Ball hit but not scored? → agent not aggressive → try aggression_bonus
├─ Opponent too hard? → add curriculum (baseline has it)
└─ Opponent too easy? → remove curriculum → try no_curriculum
```

### 12.1 Observation Variants (Siao Xuan)

| Variant | What | Risk | Config |
|---------|------|------|--------|
| **RGB** | Keep 3-channel color instead of grayscale | 3× input size, slower training | `rgb_obs` |
| **Frame difference** | Stack frame-to-frame diffs | Loses static context (court lines) | `frame_diff` |
| **Ball heatmap** | Overlay ball (x,y) as Gaussian on grayscale | Only as good as BallTracker | `heatmap_obs` |
| **Stack depth** | Test 2, 16, 32 vs current 16 frames | Trade-off: speed vs temporal info | `stack_2`, `stack_16`, `stack_32` |

### 12.2 Architecture Variants (Amber)

| Variant | What | Risk | Config |
|---------|------|------|--------|
| **TwinNatureCNN** | Separate CNN towers for policy/value | 2× CNN params, slightly slower | Code change |
| **TwinNatureCNN large** | features_dim=512 | Overfitting on small steps | Code change |
| **SpatialSoftmaxCNN** | Coordinate-aware features instead of flattened conv | Unproven in this domain | Code change |
| **Dueling architecture** | Separate advantage/value streams | More data needed | Code change |
| **Residual connections** | Skip connections in CNN towers | Marginal benefit expected | Code change |

### 12.3 Reward Shaping Variants (Yenly)

| Variant | What | Risk | Config |
|---------|------|------|--------|
| **Aggression bonus** | `+0.1` per frame ball on opponent side | May over-aggress and leave own side open | `aggression_bonus` |
| **High hit bonus** | `+1.0` per paddle contact (up from `+0.5`) | Reward hacking | `hit_bonus_high` |
| **Placement bonus** | `+0.3` for corner/far-back shots | Niche, may not generalize | Code change |
| **Combined** | Aggression + placement + hit bonus | Over-engineering risk | Code change |

### 12.4 Hyperparameter Variants (Keven)

| Parameter | Baseline | Sweep Range | Configs |
|-----------|----------|-------------|---------|
| `ent_coef` | 0.01 | `[0.0, 0.05]` loguniform | `entropy_0.0`, `entropy_0.05` |
| `batch_size` | 128 | choice(64, 128, 256) | `small_batch`, `large_batch` |
| `learning_rate` | 2.5e-4 | `[1e-4, 5e-4]` loguniform | `lr_1e-4` |
| `n_steps` | 2048 | choice(128, 256, 512, 1024) | — |
| `clip_range` | 0.2 | choice(0.1, 0.15, 0.2, 0.3) | — |
| `n_epochs` | 4 | choice(3, 4, 8) | — |

### 12.5 Training Strategy Variants (Hui Xuan)

| Variant | What | Risk | Config |
|---------|------|------|--------|
| **No curriculum** | Skip Stage 1+2, start in self-play | May never learn basic paddle control | `no_curriculum` |
| **Longer warmup** | Stage 1: 40%, Stage 2: 30%, Stage 3: 30% | Less self-play time | Code change |
| **More self-play** | Stage 1: 20%, Stage 2: 20%, Stage 3: 60% | May start self-play before ready | Code change |
| **Adaptive stages** | Stage 2 ends when win rate vs square > 80% | Complexity, may never trigger | Code change |
| **Buffer: weighted sampling** | Stronger opponents sampled more | May be too hard for learning | Code change |
| **Buffer: ELO-based** | Maintain ELO, sample proportionally | Over-engineered for 5-checkpoint pool | Code change |
