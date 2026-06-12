# Handoff — dPickleBall Bot

**Last updated:** June 12, 2026
**Status:** Code infrastructure ✅, B1 smoke test ✅, Team experiments starting now.

---

## Project Overview

A pickleball RL agent trained with PPO (SB3) for a course competition. The agent sees an `84×168` grayscale camera frame, stacks 16 frames for motion context, and outputs 3 discrete actions (vertical, horizontal, rotate).

## Current State

### What's Done
- **Code wiring** (all files exist and work)
- **B1 smoke test** (10k steps, 0 crashes) — pipeline validated
- **Bug fixes**: SmoothBallTracker wiring, paddle_y reward, eval both sides, PPO_DEFAULTS
- **Competition export** ready (`agent/competition_export.py`)
- **16-frame stack** (was 4) — frame_stack is now 16 everywhere
- **Frame_stack in sweep configs** fixed to 16

### Team Structure (5 members)

| Member | Area | Branch | Files |
|--------|------|--------|-------|
| M1 | Architecture | `team/arch-variants` | `agent/two_tower_cnn.py`, `env/wrapper.py` (preprocessing pipeline) |
| M2 | Reward | `team/reward-shaping` | `agent/reward_shaping.py`, `agent/experiment_configs.py` |
| M3 | Observation | `team/obs-variants` | `env/preprocessing.py` (BallTracker, heatmap) |
| M4 | Hyperparameters | `team/hyperparams` | `sweep_configs/*.json`, `scripts/grid_search.py` |
| M5 | Training Strategy | `team/training-strategy` | `agent/train_selfplay_v2.py`, `agent/selfplay_callback.py`, `agent/buffer.py`, `config.py` |

**Important:** `env/wrapper.py` is split — M1 owns the preprocessing pipeline (frame conversion, resize, grayscale), M3 owns ball detection + heatmap rendering consumed by the wrapper. Both coordinate via PR.

### Dependency Chain
```
M3 (Observation) produces baseline first
  └── Everyone depends on M3's baseline (heatmap_obs, stack_16)
M1, M2, M4 → work independently after M3 baseline
  └── M5 depends on checkpoints from M1, M2, M4 for Stage 3 self-play
```

## Training Plan (4-Phase Workflow)

| Phase | Steps | Purpose |
|-------|------:|---------|
| Trial experiments | 10k–20k | Quick test whether each idea helps |
| Selection | None | Compare results, no new training |
| Combination test | 50k | Test selected best parts together |
| Final training | 300k (100k × 3 stages) | Train final model with curriculum |

**Done criteria:** 50k steps minimum per variant, evaluate both left+right (50 matches each), win rate > 85% vs square = baseline met.

**Curriculum:** Stage 1 (30% no-op, dense reward) → Stage 2 (30% square, fading reward) → Stage 3 (40% self-play, sparse reward).

## Key File Reference

| File | Purpose |
|------|---------|
| `config.py` | All constants: BUILD_PATH, NUM_ENVS=8, FRAME_STACK=16, PPO defaults, curriculum splits |
| `agent/train_selfplay_v2.py` | Main training with 3-stage curriculum |
| `agent/experiment_configs.py` | Named experiment presets (parent inheritance) |
| `run_experiment.py` | Single-experiment harness |
| `scripts/grid_search.py` | Hyperparameter sweep runner |
| `sweep_configs/*.json` | Grid search configs (frame_stack=16 now) |
| `agent/wrapper_vec.py` | 8× parallel Unity env factory |
| `env/wrapper.py` | EnhancedUnityWrapper — obs pipeline + reward |
| `env/preprocessing.py` | BallTracker (HSV) |
| `agent/competition_export.py` | Final export for competition submission |

## Documentation Structure

| File | What It's For |
|------|--------------|
| `CONTRIBUTING.md` | **Start here.** Roles, setup, workflow, training plan, quick reference. |
| `PROJECT_GUIDE.md` | Technical deep dive (architecture, reward, curriculum, PPO config, experiment catalog). |
| `FLOW.md` | Architecture diagram + data shapes. |
| `DASHBOARD.md` | Milestone tracker (high-level status). |
| `PROGRESS.md` | Team progress bars. |
| `RESULTS.md` | Cross-member comparison table. |
| `progress/member-*-area.md` | Per-member experiment logs, decisions, blockers. |

## Quick Commands

```bash
# List available experiment configs
python run_experiment.py --list

# Quick test (10k steps, 2 envs)
python run_experiment.py --config heatmap_obs --steps 10000 --envs 2

# Hyperparameter sweep
python scripts/grid_search.py sweep_configs/phase3_focused.json

# Full curriculum training
python agent/train_selfplay_v2.py

# Evaluate a checkpoint
python agent/evaluate.py --checkpoint path/to/checkpoint.zip --episodes 50
```

## Key Config Values

- `FRAME_STACK = 16` (was 4)
- `NUM_ENVS = 8` (parallel Unity environments)
- `TOTAL_TIMESTEPS = 300_000` (100k per stage)
- PPO: `lr=2.5e-4`, `batch=128`, `gamma=0.99`, `ent_coef=0.01`, `n_epochs=4`
- Opponent buffer: `max_size=5`, random sampling, worst evicted

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Unity won't start | Check BUILD_PATH in config.py or .env file |
| Port conflict | `lsof -ti:5005-5012 \| xargs kill -9` |
| numpy error | `pip install "numpy<2"` |
| ML-Agents missing | `pip install -e ./dpickleball-ml-agents/ml-agents-envs` |
| Training slow | Reduce `--envs`, close other GPU apps |

## Reminders for LLM-Assisted Work

- ALL config values come from `config.py` — never hardcode
- Experiment configs inherit via `"parent": "baseline"` in `experiment_configs.py`
- Frame conversions happen in `env/wrapper.py:_preprocess()`
- Ball tracking HSV range: `H=[20,35], S=[80,255], V=[80,255]`
- Teams must evaluate BOTH left and right sides (50 matches each)
- Terminal commands must include `conda activate dpickleball` first
- Check `progress/member-*-area.md` for the member's current experiment log, blockers, and next steps
- Results go to `RESULTS.md` (cross-member comparison) + `progress/member-*-area.md` (detailed log)
- Checkpoints go to Google Drive, not GitHub (>10MB)
