# Area: Hyperparameters — Keven

## Assigned Files

| File | Role |
|------|------|
| `sweep_configs/phase3_default.json` | Default grid search config (30 trials) |
| `sweep_configs/phase3_focused.json` | Focused grid search config (20 trials) |
| `scripts/grid_search.py` | Random hyperparameter sweep script |

> **`env/wrapper.py` split awareness:** Amber owns the preprocessing pipeline (frame conversion, resize, grayscale); Siao Xuan owns ball detection + heatmap rendering. Sweep parameters depend on observation features — coordinate interface changes via PR.

---

## Current Status

**🟡 Sweep configs ready; sweeps pending**

> **⚠️ Curriculum Warning:** `run_experiment.py` (used by `grid_search.py` for sweeps) does NOT use the 3-stage curriculum (no-op → square → self-play). Sweeps train against Unity's default opponent only. Results from sweeps may **not** transfer directly to the final curriculum-based training. Use sweep results to narrow hyperparameter ranges, then validate the best configs in the full curriculum pipeline.
>
> **Base config:** Phase 4 sweeps now use `hit_aggression_placement_combo_heatmap` (heatmap + 16-frame + best reward combo from Siao Xuan + Yenly).

---

## Experiment Log

| # | Date | Config | Steps | Win Rate (right) | Win Rate (left) | Best Opponent | Drive Link | Notes |
|---|------|--------|-------|-----------------|-----------------|--------------|------------|-------|
| — | — | — | — | — | — | — | — | — |

---

## Done Criteria

These criteria apply to every experiment variant you run:

| Criterion | Standard |
|-----------|----------|
| **Minimum steps** | 50k steps per variant |
| **Evaluation sides** | Both left and right, 50 matches each |
| **Baseline met at** | Win rate > 85% vs the square opponent |

---

## Parameters to Sweep

| Parameter | Default Range | Focused Range |
|-----------|--------------|---------------|
| `learning_rate` | [1e-5, 1e-3] loguniform | [1e-4, 5e-4] loguniform |
| `ent_coef` | [0.0, 0.05] uniform | [0.005, 0.02] uniform |
| `batch_size` | choice(64, 128, 256) | choice(128, 256) |
| `n_epochs` | [3, 10] uniform int | — |
| `gamma` | [0.95, 0.999] uniform | [0.97, 0.995] uniform |
| `gae_lambda` | [0.9, 0.99] uniform | — |
| `clip_range` | [0.1, 0.3] uniform | — |

---

## Key Decisions

| Date | Decision | Rationale |
|------|----------|-----------|
| — | — | — |

---

## Blocker Log

| Blocker | Date Opened | Status | Notes |
|---------|-----------|--------|-------|
| B1 smoke test needed | — | ✅ Resolved | B1 passed 2026-06-12 (10k steps, 0 crashes) |

---

## Dependency Tracking

### Depends On (what I need from others)
| What I Need | From Who | Why | Status |
|------------|----------|-----|--------|
| Baseline observation (heatmap_obs) | **Siao Xuan** — Observation | My hyperparameter sweeps need a working observation pipeline | ✅ On main — `hit_aggression_placement_combo_heatmap` |

### Depended By (who is waiting on my results)
| Who Needs Me | What They Need | Why | Status |
|-------------|---------------|-----|--------|
| **Hui Xuan** — Training Strategy | Best hyperparameter config | Hui Xuan uses my best sweep results for the final curriculum training | ⏳ Waiting for my sweep |

---

## Next Steps

- [ ] Run `phase4_combined.json` sweep (30 trials, base: hit+aggression+placement combo with heatmap, ~6-8h on M5)
- [ ] Analyze best params from sweep → narrow ranges
- [ ] Run `phase4_focused.json` sweep with narrowed ranges (20 trials)
- [ ] Evaluate best checkpoint from each sweep via `run_eval.py` against square opponent
- [ ] Push best config row to RESULTS.md
