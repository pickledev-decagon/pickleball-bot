# Area: Observation — Member 3

## Assigned Files

| File | Role |
|------|------|
| `env/preprocessing.py` | BallTracker, SmoothBallTracker, frame_diff, paddle detection |
| `env/wrapper.py` (interface) | **M3 owns**: ball detection + heatmap rendering consumed by the wrapper. Coordinate changes with M1 via PR. |

---

## Current Status

**⏳ Pending** — waiting for B1 smoke test + repo access

---

## Experiment Log

| # | Date | Config | Steps | Win Rate (right) | Win Rate (left) | Best Opponent | Drive Link | Notes |
|---|------|--------|-------|-----------------|-----------------|--------------|------------|-------|
| — | — | — | — | — | — | — | — | — |

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

## Next Steps

- [ ] Run heatmap_obs baseline (50k steps)
- [ ] Compare stack_16 vs stack_32
- [ ] Compare heatmap vs no heatmap
- [ ] Push best config row to RESULTS.md
