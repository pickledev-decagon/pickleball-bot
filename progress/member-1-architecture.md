# Area: Model Architecture — Member 1

## Assigned Files

| File | Role |
|------|------|
| `agent/two_tower_cnn.py` | TwinNatureCNN, SpatialSoftmaxCNN, factory functions |
| `env/wrapper.py` | EnhancedUnityWrapper — **M1 owns**: preprocessing pipeline (frame conversion, resize, grayscale). **M3 owns**: ball detection + heatmap rendering consumed by the wrapper. Coordinate interface changes via PR. |

---

## Current Status

**⏳ Pending** — waiting for B1 smoke test + repo access

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

## Variants to Test

- [ ] **TwinNatureCNN** (baseline) — separate policy/value CNN towers, features_dim=256
- [ ] **TwinNatureCNN large** — features_dim=512 (risk of overfitting on small steps)
- [ ] **SpatialSoftmaxCNN** — coordinate-aware features instead of flattened conv
- [ ] **Dueling architecture** — separate advantage/value streams (if compatible with SB3)
- [ ] **Residual connections** — add skip connections in CNN towers

---

## Key Decisions

| Date | Decision | Rationale |
|------|----------|-----------|
| — | — | — |

---

## Blocker Log

| Blocker | Date Opened | Status | Notes |
|---------|-----------|--------|-------|
| B1 smoke test needed before full training | — | ✅ Resolved | B1 passed 2026-06-12 (10k steps, 0 crashes) |

---

## Next Steps

- [ ] Run TwinNatureCNN baseline (50k steps, heatmap_obs, stack_16)
- [ ] Compare SpatialSoftmaxCNN vs TwinNatureCNN
- [ ] Push best config row to RESULTS.md
