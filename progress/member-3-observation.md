# Area: Observation — Siao Xuan

## Assigned Files

| File | Role |
|------|------|
| `env/preprocessing.py` | BallTracker, SmoothBallTracker, frame_diff, paddle detection |
| `env/wrapper.py` (interface) | **Siao Xuan owns**: ball detection + heatmap rendering consumed by the wrapper. Coordinate changes with Amber via PR. |

---

## Current Status

**🟢 Ready to start**

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

- [ ] **Baseline (grayscale, no heatmap)** — plain grayscale frames
- [ ] **Heatmap** — ball position overlaid as Gaussian peaks
- [ ] **RGB** — 3-channel color input (3× channels, more info)
- [ ] **Frame difference** — motion pops out, static background removed
- [ ] **Stack_16** — 400ms temporal context (current default)
- [ ] **Stack_32** — 800ms temporal context (official example depth)
- [ ] **Heatmap + stack_32** — best observation so far?
- [ ] **SmoothBallTracker** — occlusion handling improvement (already wired)

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
| *(none — observation is the foundation layer)* | — | — | — |

### Depended By (who is waiting on my results)
| Who Needs Me | What They Need | Why | Status |
|-------------|---------------|-----|--------|
| **Amber** — Architecture | Baseline observation pipeline (heatmap_obs) | Amber needs a working observation to test architecture variants against | ⏳ Pending — I produce this first |
| **Yenly** — Reward | Baseline observation pipeline (heatmap_obs) | Yenly needs observation to test reward shaping | ⏳ Pending — I produce this first |
| **Keven** — Hyperparameters | Baseline observation pipeline (heatmap_obs) | Keven needs observation for sweep experiments | ⏳ Pending — I produce this first |
| **Hui Xuan** — Training Strategy | Baseline observation pipeline (heatmap_obs) | Hui Xuan needs observation for full curriculum training | ⏳ Pending — I produce this first |

---

## Next Steps

- [ ] **Baseline:** heatmap_obs, stack_16, 50k steps
- [ ] Compare stack_16 vs stack_32
- [ ] Compare heatmap vs no heatmap
- [ ] Push best config row to RESULTS.md
