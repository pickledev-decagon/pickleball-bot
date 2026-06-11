# Handoff — dPickleBall Bot

**Date:** May 17, 2026
**Phase:** Phase 1 ✅ Complete | Phase 2: 🟡 5/13 experiments done

---

## What Was Done (Phase 1 + Observation Variants)

### Code Changes (4 files)
| File | Lines | What |
|------|-------|------|
| `env/wrapper.py` | +90 | Added `use_rgb`, `use_frame_diff`, `use_heatmap` params; heatmap rendering with Gaussian overlay; frame_diff with thresholded absdiff; `HitDetector` + `shaped_reward` wired into `step()` |
| `agent/wrapper_vec.py` | threaded | All env_kwargs + reward_config pass through `make_unity_env_fn` → `EnhancedUnityWrapper` |
| `agent/reward_shaping.py` | added params | `aggression_bonus`, `hit_bonus`, `proximity_weight` kwargs |
| `run_experiment.py` | wired | `create_vec_env` gets `env_kwargs` and `reward_kwargs` from config resolution |

### Experiments Run (M1 Air, 2 envs, 10k steps each)

| Config | FPS | Key Metric | Verdict |
|--------|-----|------------|---------|
| baseline | 38.5 | expl.var=0.009 | New baseline (was 31.1 FPS) |
| rgb_obs | 28.8 | expl.var=0.003 | Skip — 25% slower, no benefit |
| frame_diff | 36.0 | clip_fraction=0 | Skip — near-zero frames |
| **heatmap_obs** | **33.3** | **expl.var=0.329** | **Phase 3 candidate** |
| stack_2 | 35.1 | expl.var=0.0003 | Skip — keep 4-stack |

**Heatmap is the key finding.** Explained variance jumped from 0.009→0.329 (36×). TwoTower CNN should use this as one tower.

### Bugs Fixed
- `proximity_weight` TypeError → added default + kwargs wiring
- Logger metric keys → changed to SB3 format (`entropy_loss`, `policy_gradient_loss`)

---

## What Remains

### Phase 2 — Experiments for Other Machines (8 configs left)

**Member B** (M5 Pro, 8 envs, ~20 min):
```
python run_experiment.py --config baseline     --steps 10000 --envs 8
python run_experiment.py --config entropy_0.0  --steps 10000 --envs 8
python run_experiment.py --config entropy_0.05 --steps 10000 --envs 8
python run_experiment.py --config large_batch  --steps 10000 --envs 8
python run_experiment.py --config small_batch  --steps 10000 --envs 8
python run_experiment.py --config lr_1e-4      --steps 10000 --envs 8
```

**Member C** (Laptop, 2 envs, ~15 min):
```
python run_experiment.py --config baseline         --steps 10000 --envs 2
python run_experiment.py --config aggression_bonus --steps 10000 --envs 2
python run_experiment.py --config hit_bonus_high   --steps 10000 --envs 2
```

**Member D** (Laptop, 2 envs, ~15 min):
```
python run_experiment.py --config baseline      --steps 10000 --envs 2
python run_experiment.py --config no_curriculum --steps 10000 --envs 2
```
Also: validate curriculum pipeline via `train_selfplay_v2.py`

### Phase 3 — Model Tuning (Members E+F prep, then full Phase 3)
1. **TwoTower CNN** (`agent/two_tower_cnn.py`): tower A = heatmap obs (4,84,168), tower B = game RAM features from `_get_ram_state()`, fuse at FC layer
2. **Grid search** (`scripts/grid_search.py`): sweep TwoTower variants + top Phase 2 configs
3. **LSTM** tradeoff: evaluate if temporal module adds value over 4-stack CNN

### Phase 4 — Competition Export (after best model selected)
1. TorchScript trace for <0.01s inference
2. Async inference pattern (model runs while prev action executes)
3. `competition/teamX.py` scaffolding
4. Stress test (1000 episodes, measure response time, no crashes)

---

## Key Project Files
| File | Purpose |
|------|---------|
| `run_experiment.py` | Experiment harness — `--config <name> --steps N --envs M` |
| `agent/experiment_configs.py` | 13 config presets with parent inheritance |
| `env/wrapper.py` | `EnhancedUnityWrapper` — obs pipeline + reward shaping |
| `agent/wrapper_vec.py` | 8× parallel Unity env factory |
| `agent/reward_shaping.py` | `HitDetector`, `shaped_reward`, `annealing_weight` |
| `agent/train_selfplay_v2.py` | 3-stage curriculum training (not yet run) |
| `env/preprocessing.py` | `BallTracker` HSV detection |
| `RESULTS.md` | Full experiment comparison table + JSON |
| `EXPERIMENTS.md` | Experiment catalog with hypotheses |
| `FLOW.md` | Architecture diagram + data shapes |
| `phase_2_experiments.md` | Detailed member assignments + protocol |

## Quick Start for Next Session
```bash
conda activate dpickleball
# Remaining experiments:
python run_experiment.py --config entropy_0.0 --steps 10000 --envs 8

# Phase 3 kickoff:
# 1. Create agent/two_tower_cnn.py
# 2. Create scripts/grid_search.py
# 3. Wire into experiment_configs.py with new configs
```
