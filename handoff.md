# Handoff — dPickleBall Bot

**Last updated:** June 12, 2026
**Status:** Code infrastructure ✅, B1 smoke test ✅, Team experiments starting now.

---

## 🎯 Before I Help You — Who Are You?

I need to know which area you're working on so I only give you relevant advice. Tell me your name:

- **Amber** → Model Architecture (two_tower_cnn.py, env/wrapper.py preprocessing)
- **Yenly** → Reward Shaping (reward_shaping.py, experiment_configs.py)
- **Siao Xuan** → Observation (preprocessing.py, ball detection + heatmap)
- **Keven** → Hyperparameters (sweep_configs/*.json, grid_search.py)
- **Hui Xuan** → Training Strategy (train_selfplay_v2.py, selfplay_callback.py, buffer.py, config.py)

> ⚠️ **Important:** I will ONLY give advice for YOUR assigned area. If you ask about another member's area, I'll tell you to coordinate with them instead. This prevents us from accidentally stepping on each other's work.

---

## Project Context (for everyone)

A pickleball RL agent trained with PPO (SB3) for a course competition. The agent sees an `84×168` grayscale camera frame, stacks 16 frames for motion context, and outputs 3 discrete actions (vertical, horizontal, rotate).

### What's Already Done

- All code files exist and are wired
- B1 smoke test (10k steps) passed — pipeline validated
- 4 bugs fixed: SmoothBallTracker wiring, paddle_y reward, eval both sides, PPO_DEFAULTS
- Frame stack is **16 everywhere** (was 4)
- Competition export ready

### Key Shared Configs

| Setting | Value |
|---------|-------|
| Frame stack | `16` |
| Parallel envs | `8` |
| Total timesteps | `300_000` (100k per curriculum stage) |
| PPO base | `lr=2.5e-4`, `batch=128`, `gamma=0.99`, `ent_coef=0.01`, `n_epochs=4` |
| Opponent buffer | `max_size=5`, random sampling, worst evicted |

### Training Plan (4 Phases)

| Phase | Steps | What you do |
|-------|------:|-------------|
| Trial experiments | 10k–20k | Quick test to see if an idea helps |
| Selection | No training | Compare results, pick the best |
| Combination test | 50k | Run your best variant at full length, **evaluate both sides** (50 matches each), **85% vs square = baseline met** |
| Final training | 300k | Hui Xuan runs this after getting everyone's best checkpoints |

### Dependency Chain

```
Siao Xuan (Observation) produces baseline first
  └── Amber, Yenly, Keven need Siao Xuan's baseline to test
  └── Hui Xuan runs final training after getting checkpoints from Amber, Yenly, Keven
```

### Shared File Ownership

`env/wrapper.py` is **split ownership**:
- **Amber** owns: preprocessing pipeline (frame conversion, resize, grayscale)
- **Siao Xuan** owns: ball detection + heatmap rendering consumed by the wrapper
- Coordinate changes via PR

---

<!-- ============================================================ -->
<!-- ONLY SHOW THE SECTION BELOW THAT MATCHES THE USER'S AREA     -->
<!-- ============================================================ -->

## If you are Amber — Model Architecture

### Your Files

| File | What it does |
|------|-------------|
| `agent/two_tower_cnn.py` | TwinNatureCNN, SpatialSoftmaxCNN, factory functions |
| `env/wrapper.py` | YOU own: frame conversion, resize, grayscale pipeline. SIAO XUAN owns: ball detection, heatmap rendering. Coordinate via PR. |

### Your Experiment Variants

| Priority | Variant | Description | Config |
|----------|---------|-------------|--------|
| **1** | **Baseline (TwinNatureCNN)** | Separate policy/value CNN towers, features_dim=256 | Code change |
| 2 | TwinNatureCNN large | features_dim=512 (overfitting risk) | Code change |
| 3 | SpatialSoftmaxCNN | Coordinate-aware features instead of flattened conv | Code change |
| 4 | Dueling architecture | Separate advantage/value streams | Code change |
| 5 | Residual connections | Skip connections in CNN towers | Code change |

### What to Run First

```bash
conda activate dpickleball

# 1. Wait for Siao Xuan's baseline observation
# 2. Run your baseline variant:
python run_experiment.py --config heatmap_obs --steps 10000 --envs 2
# 3. After quick test passes, run combination test (50k):
python run_experiment.py --config heatmap_obs --steps 50000 --envs 8
# 4. Evaluate:
python agent/evaluate.py --checkpoint path/to/checkpoint.zip --episodes 50
```

**Do NOT** modify `env/preprocessing.py` (Siao Xuan's territory), `config.py` (Hui Xuan's territory), or reward files (Yenly's territory).

### Your Dependencies

| You need | From | Status |
|----------|------|--------|
| Baseline observation (heatmap_obs, stack_16) | Siao Xuan | ⏳ Waiting |

| Who needs you | What they need | Status |
|--------------|---------------|--------|
| Hui Xuan | Your best architecture checkpoint | ⏳ Waiting for you |

---

## If you are Yenly — Reward Shaping

### Your Files

| File | What it does |
|------|-------------|
| `agent/reward_shaping.py` | proximity, hit detection, placement, aggression rewards |
| `agent/experiment_configs.py` | Named experiment presets (reward configs) |

### Your Experiment Variants

| Priority | Variant | Config | What to change |
|----------|---------|--------|----------------|
| **1** | **Baseline** | `--config baseline` | proximity_weight=1.0, hit_bonus=0.5, aggression=0.0, placement=0.0 |
| 2 | Aggression bonus | `--config aggression_bonus` | +0.1 per frame ball on opponent side |
| 3 | High hit bonus | `--config hit_bonus_high` | hit_bonus=1.0 (up from 0.5) |
| 4 | Placement bonus | Code change | +0.3 for corner/far-back shots |
| 5 | Combined | Code change | aggression + placement + hit bonus |

### What to Run First

```bash
conda activate dpickleball

# 1. Wait for Siao Xuan's baseline observation
# 2. Run your baseline:
python run_experiment.py --config baseline --steps 10000 --envs 2
# 3. Test variants:
python run_experiment.py --config aggression_bonus --steps 10000 --envs 2
python run_experiment.py --config hit_bonus_high --steps 10000 --envs 2
# 4. Run your best at 50k, evaluate both sides
```

**Do NOT** modify `env/wrapper.py` or `env/preprocessing.py` (observation is Amber/Siao Xuan's territory). If you need a new observation feature, open a PR.

### Your Dependencies

| You need | From | Status |
|----------|------|--------|
| Baseline observation (heatmap_obs, stack_16) | Siao Xuan | ⏳ Waiting |

| Who needs you | What they need | Status |
|--------------|---------------|--------|
| Hui Xuan | Your best reward checkpoint | ⏳ Waiting for you |

---

## If you are Siao Xuan — Observation

### Your Files

| File | What it does |
|------|-------------|
| `env/preprocessing.py` | BallTracker, SmoothBallTracker, frame_diff, paddle detection |
| `env/wrapper.py` (interface) | YOU own: ball detection + heatmap rendering. AMBER owns: preprocessing pipeline. Coordinate via PR. |

### Your Experiment Variants

| Priority | Variant | Config | What it does |
|----------|---------|--------|-------------|
| **1** | **Heatmap baseline** | `--config heatmap_obs` | Ball position overlaid as Gaussian peaks on grayscale — **everyone depends on this** |
| 2 | Grayscale (no heatmap) | `--config baseline` | Plain grayscale frames |
| 3 | RGB | `--config rgb_obs` | 3-channel color (3× channels) |
| 4 | Frame difference | `--config frame_diff` | Motion pops out, static bg removed |
| 5 | Stack_32 | `--config stack_32` | 800ms temporal context |
| 6 | Heatmap + stack_32 | Code change | Best observation candidate |
| 7 | SmoothBallTracker | Already wired | Occlusion handling improvement |

### What to Run First

```bash
conda activate dpickleball

# ⚠️ YOU ARE FIRST. Everyone waits for your baseline.
# Run the heatmap baseline immediately:
python run_experiment.py --config heatmap_obs --steps 10000 --envs 2
python run_experiment.py --config heatmap_obs --steps 50000 --envs 8

# Then explore variants:
python run_experiment.py --config baseline --steps 10000 --envs 2
python run_experiment.py --config rgb_obs --steps 10000 --envs 2
python run_experiment.py --config stack_32 --steps 10000 --envs 2
```

**Do NOT** modify `agent/two_tower_cnn.py` (Amber's territory), `agent/reward_shaping.py` (Yenly's territory), or `config.py` (Hui Xuan's territory).

### Your Dependencies

| You need | From | Status |
|----------|------|--------|
| *(none — you're the foundation layer)* | — | — |

| Who needs you | What they need | Status |
|--------------|---------------|--------|
| Amber | Baseline observation (heatmap_obs) | ⏳ You produce this first |
| Yenly | Baseline observation (heatmap_obs) | ⏳ You produce this first |
| Keven | Baseline observation (heatmap_obs) | ⏳ You produce this first |
| Hui Xuan | Baseline observation (heatmap_obs) | ⏳ You produce this first |

---

## If you are Keven — Hyperparameters

### Your Files

| File | What it does |
|------|-------------|
| `sweep_configs/phase3_default.json` | Default grid search (30 trials, frame_stack=16) |
| `sweep_configs/phase3_focused.json` | Focused grid search (20 trials, frame_stack=16) |
| `scripts/grid_search.py` | Random hyperparameter sweep runner |

### Your Sweep Parameters

| Parameter | Default Range | Focused Range |
|-----------|--------------|---------------|
| `learning_rate` | [1e-5, 1e-3] loguniform | [1e-4, 5e-4] loguniform |
| `ent_coef` | [0.0, 0.05] uniform | [0.005, 0.02] uniform |
| `batch_size` | choice(64, 128, 256) | choice(128, 256) |
| `n_epochs` | [3, 10] uniform int | — |
| `gamma` | [0.95, 0.999] uniform | [0.97, 0.995] uniform |
| `gae_lambda` | [0.9, 0.99] uniform | — |
| `clip_range` | [0.1, 0.3] uniform | — |

### What to Run First

```bash
conda activate dpickleball

# ⚠️ Wait for Siao Xuan's baseline first.
# Then run sweeps:
python scripts/grid_search.py sweep_configs/phase3_default.json --output results/sweeps
python scripts/grid_search.py sweep_configs/phase3_focused.json --output results/sweeps
```

⚠️ **Curriculum warning:** `run_experiment.py` (used by grid_search) does NOT use the 3-stage curriculum. Sweeps train against Unity's default opponent. Results may not transfer directly to Hui Xuan's final curriculum training. Use sweeps to narrow ranges, then validate best configs in the full pipeline.

**Do NOT** modify `env/wrapper.py`, `env/preprocessing.py`, `agent/two_tower_cnn.py`, or `agent/reward_shaping.py` — those belong to other members.

### Your Dependencies

| You need | From | Status |
|----------|------|--------|
| Baseline observation (heatmap_obs, stack_16) | Siao Xuan | ⏳ Waiting |

| Who needs you | What they need | Status |
|--------------|---------------|--------|
| Hui Xuan | Best hyperparameter config | ⏳ Waiting for your sweep |

---

## If you are Hui Xuan — Training Strategy

### Your Files

| File | What it does |
|------|-------------|
| `agent/train_selfplay_v2.py` | Main training script with 3-stage curriculum |
| `agent/selfplay_callback.py` | Mid-training checkpoint → buffer logic |
| `agent/buffer.py` | OpponentBuffer (sampling, eviction) |
| `config.py` | Central config (all constants) |

### Your Experiment Variants

| Priority | Variant | What to change |
|----------|---------|----------------|
| **1** | **Default curriculum** | Stage 1: 30%, Stage 2: 30%, Stage 3: 40% (already set) |
| 2 | Longer warmup | Stage 1: 40%, Stage 2: 30%, Stage 3: 30% |
| 3 | More self-play | Stage 1: 20%, Stage 2: 20%, Stage 3: 60% |
| 4 | Adaptive | Stage 2 ends when win rate vs square > 80% |
| 5 | Weighted sampling | Stronger opponents sampled more |
| 6 | ELO-based | Maintain ELO, sample proportionally |

### What to Run First

```bash
conda activate dpickleball

# Stage 1+2 can start immediately:
# Stage 3 needs checkpoints from Amber, Yenly, and Keven.
# Run default curriculum when ready:
python agent/train_selfplay_v2.py
```

### Your Dependencies

| You need | From | Status |
|----------|------|--------|
| Baseline observation (heatmap_obs, stack_16) | Siao Xuan | ⏳ Waiting |
| Architecture checkpoint | Amber | ⏳ Waiting |
| Reward config checkpoint | Yenly | ⏳ Waiting |
| Best hyperparameters | Keven | ⏳ Waiting |

| Who needs you | What they need | Status |
|--------------|---------------|--------|
| *(none — you produce the final model)* | — | — |

---

## Universal Reminders (Everyone)

- Always `conda activate dpickleball` before running commands
- All config values come from `config.py` — never hardcode
- Experiment configs inherit via `"parent": "baseline"` in `experiment_configs.py`
- Ball tracking HSV range: `H=[20,35], S=[80,255], V=[80,255]` (don't change)
- Evaluate BOTH left and right sides, 50 matches each
- Checkpoints >10MB go to Google Drive, NOT GitHub
- Code changes go on your branch, open a PR to main
- Log all results in `progress/member-X-AREA.md` and `RESULTS.md`
- If you're stuck on something outside your area → ask the relevant member, don't try to fix it yourself
- If you're stuck on something in your area → I can help, but first check if a similar variant already exists in `agent/experiment_configs.py` (`python run_experiment.py --list`)
