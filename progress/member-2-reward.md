# Area: Reward Functions — Yenly

## Assigned Files

| File | Role |
|------|------|
| `agent/reward_shaping.py` | proximity, hit detection, placement, aggression rewards |
| `agent/experiment_configs.py` | Named experiment presets (reward configs) |

> **`env/wrapper.py` split awareness:** Amber owns the preprocessing pipeline (frame conversion, resize, grayscale); Siao Xuan owns ball detection + heatmap rendering. If your reward config needs new observation features, coordinate via PR with the relevant owner.

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

- [ ] **Baseline** — proximity_weight=1.0, hit_bonus=0.5, aggression=0.0, placement=0.0
- [ ] **Aggression bonus** — +0.1 per frame ball on opponent side
- [ ] **High hit bonus** — hit_bonus=1.0 (reinforce hitting more)
- [ ] **Placement bonus** — +0.3 for corner/far-back shots
- [ ] **Combined** — aggression + placement + hit bonus
- [ ] **Paddle-aware proximity** — if `detect_paddles` is wired, use actual paddle position

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
| Baseline observation (heatmap_obs) | **Siao Xuan** — Observation | My reward variants need a working observation pipeline to test against | ⏳ Waiting for Siao Xuan baseline |

### Depended By (who is waiting on my results)
| Who Needs Me | What They Need | Why | Status |
|-------------|---------------|-----|--------|
| **Hui Xuan** — Training Strategy | Trained reward checkpoint | Hui Xuan's self-play needs my best reward config checkpoint | ⏳ Waiting for my baseline |

---

## Next Steps

- [ ] **Baseline:** heatmap_obs, stack_16, 50k steps
- [ ] Run aggression bonus variant
- [ ] Run placement bonus variant
- [ ] Run combined variant
- [ ] Push best config row to RESULTS.md
