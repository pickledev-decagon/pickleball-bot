# dPickleBall Bot — Project Dashboard

**Generated:** June 12, 2026
**Code-only work:** 100% complete
**Training runs:** 1 (B1 smoke test passed ✅)

---

## 🚦 Complete Milestone Tracker

| # | Milestone | Status | When | Files | Time |
|---|-----------|--------|------|-------|------|
| **P0** | Environment setup | ✅ Done | May 4 | `dpickleball-ml-agents/`, conda env | — |
| **P1** | Ball tracking (HSV) | ✅ Done | May 4 | `env/preprocessing.py` | — |
| **P2** | Self-play v2 pipeline | ✅ Done | May 4–13 | `train_selfplay_v2.py`, wrapper | — |
| | _Phase 2 experiments (heatmap winner)_ | ✅ Done | May 13 | `experiment_configs.py`, `RESULTS.md` | — |
| | _TwoTower CNN + grid search scripts_ | ✅ Done | May 20 | `two_tower_cnn.py`, `grid_search.py` | — |
| **F1** | Left-side training (obs flip + mirror) | ✅ **Done** | May 20 | `env/wrapper.py`, `agent/wrapper_vec.py` | 🔴 CRITICAL |
| **F2** | Async inference (0.01s deadline) | ✅ **Done** | May 20 | `agent/competition_export.py` | 🔴 CRITICAL |
| **F3** | 16-frame stack (was 4) | ✅ **Done** | May 20 | `wrapper.py`, `wrapper_vec.py`, `configs` | 🟡 HIGH |
| **F4** | Win-rate evaluation harness | ✅ **Done** | May 20 | `agent/eval_harness.py` | 🟡 HIGH |
| **F5** | Shot placement reward + rotation | ✅ **Done** | May 20 | `agent/reward_shaping.py` | 🟡 MEDIUM |
| **F6** | Ball tracker occlusion handling | ✅ **Done** | May 20 | `env/preprocessing.py` | 🟡 MEDIUM |
| **A1** | Tracker validation (real Unity frames) | ✅ **Passed** | May 20 | `scripts/validate_tracker.py` | 100% detection |
| **A2** | Competition export dry-run | ✅ **Passed** | May 20 | `scripts/test_competition_export.py` | <2ms response |
| **C1** | Mid-training checkpoint → buffer | ✅ **Done** | May 20 | `agent/selfplay_callback.py` | 🟡 HIGH |
| **C2** | Centralized config file | ✅ **Done** | May 20 | `config.py` (new) | 🟢 LOW |
| **C3** | Fixed legacy evaluate.py | ✅ **Done** | May 20 | `agent/evaluate.py` | 🟢 LOW |
| **C4** | Updated competition docs | ✅ **Done** | May 20 | `phase_4_competition_export.md` | 🟢 LOW |

---

## 📋 Remaining Tasks

| Task | Est. Time | Unlocks | 
|------|-----------|---------|
| **✅ B1: 10k-step smoke test** | **5 min** | **_Gate passed — pipeline validated_** |
| **B2: First 50k–100k training run** | 30–60 min | First checkpoint + baseline win rates |
| **B3: Self-play with populated buffer** | During B2 | Agent improves by playing itself |
| **B4: Grid search (sweep)** | 2–4 hrs | Optimal hyperparameters |
| **B5: Offline win-rate buffer eval** | 30 min/ckpt | Accurate opponent ranking |
| **S1: Create teamX.py / teamY.py** | 30 min | Competition file structure |
| **S2: Package submission zip** | 15 min | Strip training artifacts |
| **S3: Stress test (1000 calls)** | 10 min | Verify 0% crash rate |

---

## 📈 Time to Competition

```
MINIMUM VIABLE ──▶  B2 (1h) → S1+S2 (45m)  =  ~2 hours
                         ↓
                    First real checkpoint + win rates

COMPETITIVE ──────▶  B2 (1h) → B4 (3h) → B5 (1h) → S1+S2 (45m)  =  ~6 hours
                         ↓
                    Tuned hyperparameters + strong baseline

ALL-IN ───────────▶  Full plan above + B3 iteration loop  =  ~6–8 hours
```

---

## 🧩 Where All Effort Has Gone

```
Code & Architecture Work  ████████████████████████████████  75%  (all 6 gaps + cleanup)
Validation Testing        ██████                            10%  (A1 + A2 + B1 passed)
Training Runs              ████                              8%  (B1: 10k steps, 0 crashes)
Competition Packaging      ██                                4%  (planned)
Buffer/Unused Code         ██                                3%  (legacy / archive)
```

## 🔴 Key Risk

**B1 gate is passed — the pipeline works end-to-end.** The key risk now is team coordination: merging 5 members' independent experiments into a coherent final model. Use PROGRESS.md + RESULTS.md to track progress across all areas.
