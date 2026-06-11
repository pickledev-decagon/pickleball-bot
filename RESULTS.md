# Experiment Results

Copy your experiment JSON output into the sections below and update the comparison table.

---

## Quick Comparison

| Config | Steps | FPS | Entropy | Value Loss | Expl. Var. | Machine | Verdict |
|---|---|---|---|---|---|---|---|---|
| baseline | 10000 | 38.5 | -3.29 | 0.0294 | 0.009 | M1 Air | baseline |
| rgb_obs | 10000 | 28.8 | -3.29 | 0.0296 | 0.003 | M1 Air | 25% slower |
| frame_diff | 10000 | 36.0 | -3.30 | 0.0338 | 0.000 | M1 Air | weak signal |
| heatmap_obs | 10000 | 33.3 | -3.29 | 0.0293 | 0.329 | M1 Air | best expl.var |
| stack_2 | 10000 | 35.1 | -3.28 | 0.0338 | 0.0003 | M1 Air | similar to bl |
| entropy_0.0 | — | — | — | — | — | — | |
| entropy_0.05 | — | — | — | — | — | — | |
| aggression_bonus | — | — | — | — | — | — | |
| no_curriculum | — | — | — | — | — | — | |
| large_batch | — | — | — | — | — | — | |
| small_batch | — | — | — | — | — | — | |
| hit_bonus_high | — | — | — | — | — | — | |
| lr_1e-4 | — | — | — | — | — | — | |

---

## Full Results

### baseline (Phase 1 — 10k steps, rewired stack)

```
{
  "experiment": "baseline",
  "description": "Current default: grayscale, 4-stack, entropy 0.01, curriculum, proximity reward",
  "timestamp": "20260516_235640",
  "total_steps": 10000,
  "num_envs": 2,
  "elapsed_seconds": 259.6,
  "steps_per_second": 38.5,
  "train/value_loss": 0.0294,
  "train/approx_kl": 7.30e-5
}
```

Ran on MacBook Air M1 (8GB), 2 parallel envs, 10000 steps.

**Interpretation:**
- FPS improvement: 38.5 (vs previous 31.1). Phase 1 wiring didn't add overhead.
- `value_loss = 0.029` → lower than before (0.165), critic converging better.
- `entropy = -3.29` → max exploration maintained.
- New baseline for comparison.

### rgb_obs

```
{
  "experiment": "rgb_obs",
  "description": "3-channel RGB input instead of grayscale — more color info, 3x channels",
  "timestamp": "20260517_000145",
  "total_steps": 10000,
  "num_envs": 2,
  "elapsed_seconds": 347.7,
  "steps_per_second": 28.8,
  "train/entropy_loss": -3.28,
  "train/policy_gradient_loss": -0.00160,
  "train/value_loss": 0.0296,
  "train/approx_kl": 0.00783
}
```

**Interpretation:**
- FPS dropped ~25% (28.8 vs 38.5) — 3× channels = more data to process.
- Metrics otherwise very similar to baseline. No clear benefit at 10k steps.
- **Verdict:** Skip for Phase 3 (too slow, no signal improvement).

### frame_diff

```
{
  "experiment": "frame_diff",
  "description": "Stack frame-to-frame differences instead of raw frames — motion pops out",
  "timestamp": "20260517_000757",
  "total_steps": 10000,
  "num_envs": 2,
  "elapsed_seconds": 277.7,
  "steps_per_second": 36.0,
  "train/entropy_loss": -3.30,
  "train/policy_gradient_loss": -0.000127,
  "train/value_loss": 0.0338,
  "train/approx_kl": 1.24e-05
}
```

**Interpretation:**
- `approx_kl = 1.24e-05` and `clip_fraction = 0` for all iterations → near-zero policy updates.
- Frame diffs are mostly zero (ball moves slowly), so the CNN sees blank inputs.
- `value_loss = 0.0338` slightly higher than baseline.
- **Verdict:** Skip — observation representation needs work (threshold tuning, multi-frame accumulation).

### heatmap_obs

```
{
  "experiment": "heatmap_obs",
  "description": "Ball position (x,y) overlaid as Gaussian peaks on stacked grayscale",
  "timestamp": "20260517_001300",
  "total_steps": 10000,
  "num_envs": 2,
  "elapsed_seconds": 300.3,
  "steps_per_second": 33.3,
  "train/entropy_loss": -3.29,
  "train/policy_gradient_loss": -0.00206,
  "train/value_loss": 0.0293,
  "train/approx_kl": 0.00623
}
```

**Interpretation:**
- `explained_variance = 0.329` — dramatically higher than baseline (0.009). The value function actually learns with heatmap input.
- Metrics otherwise similar to baseline, but the expl.var. jump suggests the agent gets a much clearer state signal.
- FPS only 13% slower than baseline (33.3 vs 38.5).
- **Verdict:** Strong candidate for Phase 3. The ball position Gaussian gives the critic a usable signal.

### stack_2

```
{
  "experiment": "stack_2",
  "description": "Only 2 stacked frames — faster inference, less temporal info",
  "timestamp": "20260517_001832",
  "total_steps": 10000,
  "num_envs": 2,
  "elapsed_seconds": 284.8,
  "steps_per_second": 35.1,
  "train/entropy_loss": -3.28,
  "train/policy_gradient_loss": -0.00319,
  "train/value_loss": 0.0338,
  "train/approx_kl": 0.00912
}
```

**Interpretation:**
- FPS similar to baseline (35.1 vs 38.5). Minor improvement from fewer frames.
- `explained_variance = 0.0003` — essentially zero (less temporal context hurts).
- Metrics suggest 4-stack provides useful temporal info that 2-stack lacks.
- **Verdict:** Keep 4-stack as default.

### entropy_0.0

```
[paste JSON here]
```

### entropy_0.05

```
[paste JSON here]
```

### aggression_bonus

```
[paste JSON here]
```

### no_curriculum

```
[paste JSON here]
```

### large_batch

```
[paste JSON here]
```

### small_batch

```
[paste JSON here]
```

### hit_bonus_high

```
[paste JSON here]
```

### lr_1e-4

```
[paste JSON here]
```

---

## Winner Selection

After all experiments complete, top 2 by combination of:
1. **ep_rew_mean trend** (positive slope, final value)
2. **entropy** (still exploring at end = didn't collapse)
3. **FPS** (fast enough for competition 0.01s limit)

Selected configs move to medium validation (50k steps with curriculum + 5k eval episodes). Final winner gets exported to `competition/teamX.py`.
