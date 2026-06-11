# Experiment Catalog

Every approach we try, with hypothesis, risk, validation metric, and current results. After each run, add data to [RESULTS.md](RESULTS.md).

---

## Metrics Interpretation Primer

Before running experiments, understand what the metrics mean:

| Metric | What It Measures | Good Sign | Bad Sign |
|---|---|---|---|
| `entropy` | Policy randomness. Starts at `-3.3` (max entropy for 3×3 actions). | `-3.5` to `-2.5` (slowly decreasing) | `> -1.5` = collapsed, stuck; `< -3.5` = still random |
| `approx_kl` | How much the policy changed in this update | `< 0.01` (stable) | `> 0.02` (destabilizing) |
| `value_loss` | Critic prediction error | Decreasing | Increasing (critic diverging) |
| `explained_variance` | How well critic explains returns | Increasing toward `1.0` | Stuck at `0.0` (critic learning nothing) |
| `policy_loss` | Policy gradient magnitude | Noisy, non-zero | Constant zero (policy not updating) |
| `fps` | Training speed | Consistent | Dropping (bottleneck) |
| `ep_rew_mean` | Avg episode return | Trending up | Flat |
| `ep_len_mean` | Avg episode length | Decreasing (faster points) | Increasing (longer rallies) |

### Decision Tree for Next Experiment

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

Observing results, want more speed?
├─ Try stack_2 (half the input)
└─ Try small_batch (faster updates)

Observing results, want more stability?
├─ Try large_batch (smoother gradients)
└─ Try lr_1e-4 (slower but safer)
```

---

## 1. Observation Variants

### 1.1 RGB Observation

| | |
|---|---|
| **What** | Keep 3-channel RGB instead of converting to grayscale |
| **Why** | Color may help distinguish ball, paddles, court lines |
| **Risk** | 3× more channels = 3× input size (12→36 channels with stack), slower training, more params |
| **Config** | `--config rgb_obs` |
| **Hypothesis** | `rgb_obs` improves ball proximity but reduces FPS by ~2× |

### 1.2 Frame Difference

| | |
|---|---|
| **What** | Stack frame-to-frame differences instead of raw frames |
| **Why** | Motion energy highlights the ball trajectory. Ball is the fastest-moving object on court. |
| **Risk** | Static context (court lines, net) is lost — agent can't learn court geometry |
| **Config** | `--config frame_diff` |
| **Hypothesis** | Ball proximity improves; agent may struggle with positioning |

### 1.3 Ball Heatmap Overlay

| | |
|---|---|
| **What** | Overlay ball (x,y) as 2D Gaussian peak on each grayscale frame in the stack |
| **Why** | CNN sees exactly where the ball is — no need to learn ball detection from pixels |
| **Risk** | Only as good as BallTracker. If tracking fails, heatmap lies. Mitigation: fall back to blank when `ball_pos=(-1,-1)`. |
| **Config** | `--config heatmap_obs` |
| **Hypothesis** | Best ball proximity, best reward slope — CNN spends zero capacity on detection |

### 1.4 Different Frame Stack Depth

| | |
|---|---|
| **What** | Test 2 frames vs current 4 |
| **Why** | 2 = faster inference (half input channels). 4 = current. |
| **Risk** | 2 may not capture enough motion for trajectory prediction |
| **Config** | `--config stack_2` |
| **Hypothesis** | `stack_2` is 2× faster but ~equivalent performance (paddle moves slowly) |

---

## 2. Architecture Variants

⚠️ These require code changes — not runnable via `run_experiment.py` yet.

### 2.1 Dueling Network Head

Split CNN features into separate streams: state-value `V(s)` and advantage `A(s,a)`. `Q(s,a) = V(s) + A(s,a) - mean(A)`. Helps when many states have similar action values (proven in Atari sparse-reward games).

**Risk:** More parameters = more data needed. May not show benefit at 10k steps.

### 2.2 NatureCNN (SB3 default)

SB3's built-in NatureCNN has different kernel sizes than our CustomCNN. Minimal expected difference.

### 2.3 Deeper CNN

4 conv layers (32→64→128→128) instead of 3 (32→64→64). Higher capacity, higher overfit risk.

---

## 3. Reward Shaping Variants

### 3.1 Aggression Bonus

| | |
|---|---|
| **What** | `+0.1` per timestep when ball is on opponent's side of the court |
| **Why** | Encourages agent to hit the ball toward opponent rather than dinking/sitting back |
| **Risk** | May over-aggress and leave own side open (overcommit) |
| **Config** | `--config aggression_bonus` |
| **Metric** | Points scored per episode, hit count |

### 3.2 Paddle Hit Bonus (high)

| | |
|---|---|
| **What** | `+1.0` per detected paddle contact (up from `+0.5` baseline) |
| **Why** | Stronger reinforcement for making contact with the ball |
| **Risk** | Reward hacking — agent may spaz paddle to rack up hit rewards without actually scoring |
| **Config** | `--config hit_bonus_high` |
| **Metric** | Hit count vs points scored. If hits increase but points don't → reward hacking. |

### 3.3 Positioning Penalty

Already the baseline `dense_distance_reward = -dist * 0.01`. Test by adjusting `proximity_weight` in `reward_kwargs`.

---

## 4. Hyperparameter Variants

### 4.1 Entropy Coefficient

Controls exploration vs exploitation:

| Config | Value | Expected |
|---|---|---|
| `entropy_0.0` | 0.0 | Less exploration. May converge faster or get stuck in local optimum. |
| `baseline` | 0.01 | Moderate exploration. SB3 Atari standard. |
| `entropy_0.05` | 0.05 | High exploration. More random actions, slower to converge, may find better strategies. |

### 4.2 Batch Size

| Config | Value | Expected |
|---|---|---|
| `small_batch` | 64 | Noisier gradients, more updates per second |
| `baseline` | 128 | Good balance |
| `large_batch` | 256 | Smoother gradients, fewer updates per second |

### 4.3 Learning Rate

| Config | Value | Expected |
|---|---|---|
| `baseline` | 2.5e-4 | Default — good for most environments |
| `lr_1e-4` | 1e-4 | Slower, more stable. Helpful if approx_kl is high. |

---

## 5. Training Strategy Variants

### 5.1 No Curriculum

| | |
|---|---|
| **What** | Skip Stage 1 (no-op) and Stage 2 (square-motion). Start directly in self-play. |
| **Why** | Maybe curriculum adds no value — agent can learn from self-play immediately |
| **Risk** | Agent may never learn basic paddle control if opponent is too good |
| **Config** | `--config no_curriculum` |
| **Metric** | ep_rew_mean slope, entropy curve |

### 5.2 Opponent Buffer Size

Test buffer size 3 vs 8 (baseline is 5). Manual code change needed. More opponents = better generalization; fewer = simpler, faster eviction of weak checkpoints.

---

## Quick Command Reference

```bash
# List all experiments
python run_experiment.py --list

# Run single experiment
python run_experiment.py --config <name> --steps 10000 --envs 4

# After each run, paste JSON into RESULTS.md
```

| Config | Quick Why |
|---|---|
| `baseline` | Reference — everything's default |
| `entropy_0.0` | Does entropy help or hurt? |
| `entropy_0.05` | Find optimal exploration level |
| `rgb_obs` | Does color information matter? |
| `frame_diff` | Does explicit motion help? |
| `heatmap_obs` | Tell CNN exactly where ball is |
| `aggression_bonus` | Encourage scoring behavior |
| `no_curriculum` | Is slow ramp-up necessary? |
| `large_batch` | Stable gradients |
| `small_batch` | Fast iteration |
| `hit_bonus_high` | Reinforce ball contact |
| `lr_1e-4` | Stability vs speed trade-off |
| `stack_2` | Half the obs, double the speed |
