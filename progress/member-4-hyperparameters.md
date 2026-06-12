# Area: Hyperparameters — Member 4

## Assigned Files

| File | Role |
|------|------|
| `sweep_configs/phase3_default.json` | Default grid search config (30 trials) |
| `sweep_configs/phase3_focused.json` | Focused grid search config (20 trials) |
| `scripts/grid_search.py` | Random hyperparameter sweep script |

> **`env/wrapper.py` split awareness:** M1 owns the preprocessing pipeline (frame conversion, resize, grayscale); M3 owns ball detection + heatmap rendering. Sweep parameters depend on observation features — coordinate interface changes via PR.

---

## Current Status

**⏳ Pending** — waiting for B1 smoke test + repo access

> **⚠️ Curriculum Warning:** `run_experiment.py` (used by `grid_search.py` for sweeps) does NOT use the 3-stage curriculum (no-op → square → self-play). Sweeps train against Unity's default opponent only. Results from sweeps may **not** transfer directly to the final curriculum-based training. Use sweep results to narrow hyperparameter ranges, then validate the best configs in the full curriculum pipeline.

---

## Experiment Log

| # | Date | Config | Steps | Best Win Rate (right) | Best Win Rate (left) | Key Finding | Drive Link |
|---|------|--------|-------|----------------------|---------------------|-------------|------------|
| — | — | — | — | — | — | — | — |

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

## Next Steps

- [ ] Run `phase3_default.json` sweep (30 trials, ~X hours)
- [ ] Analyze best params from sweep
- [ ] Run `phase3_focused.json` sweep with narrowed ranges (20 trials)
- [ ] Push best config row to RESULTS.md
