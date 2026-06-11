# Area: Reward Functions — Member 2

## Assigned Files

| File | Role |
|------|------|
| `agent/reward_shaping.py` | proximity, hit detection, placement, aggression rewards |
| `agent/experiment_configs.py` | Named experiment presets (reward configs) |

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
| B1 smoke test needed | — | ⏳ Blocked | — |

---

## Next Steps

- [ ] Run baseline (50k steps, heatmap_obs, stack_16)
- [ ] Run aggression bonus variant
- [ ] Run placement bonus variant
- [ ] Run combined variant
- [ ] Push best config row to RESULTS.md
