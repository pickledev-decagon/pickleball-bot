# Area: Training Strategy — Member 5

## Assigned Files

| File | Role |
|------|------|
| `agent/train_selfplay_v2.py` | Main training script with curriculum |
| `agent/selfplay_callback.py` | Mid-training checkpoint → buffer logic |
| `agent/buffer.py` | OpponentBuffer (sampling, eviction) |
| `config.py` | Central config (training constants) |

> **`env/wrapper.py` split awareness:** M1 owns the preprocessing pipeline (frame conversion, resize, grayscale); M3 owns ball detection + heatmap rendering. Final training uses all members' components — coordinate integration issues via PR.

---

## Current Status

**🟢 Ready to start**

> **📦 Checkpoint Dependency:** Stage 3 (self-play) of the curriculum requires trained checkpoints from M1–M4 to populate the opponent buffer. You can start with Stage 1+2 experiments (no-op → square opponents) while waiting for teammates to produce their checkpoints. Coordinate with members via shared Google Drive folder.

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

### Curriculum Stage Splits
- [ ] **Default** — Stage 1: 30%, Stage 2: 30%, Stage 3: 40%
- [ ] **Longer warmup** — Stage 1: 40%, Stage 2: 30%, Stage 3: 30%
- [ ] **More self-play** — Stage 1: 20%, Stage 2: 20%, Stage 3: 60%
- [ ] **Adaptive** — Stage 2 ends when win rate vs square > 80%

### Opponent Buffer Strategy
- [ ] **Random sampling** (current) — equal chance for all checkpoints
- [ ] **Weighted by win rate** — stronger opponents sampled more
- [ ] **ELO-based** — maintain ELO ratings, sample proportionally
- [ ] **Curriculum within self-play** — start with weakest, progress to strongest

### Training Duration
- [ ] **300k steps** (current default)
- [ ] **500k steps** — longer training
- [ ] **1M steps** — full convergence

### Checkpoint Frequency
- [ ] Every 10% (current — 10 checkpoints)
- [ ] Every 5% (20 checkpoints, more buffer diversity)
- [ ] Every 25% (4 checkpoints, less overhead)

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

- [ ] **Baseline:** heatmap_obs, stack_16, 300k steps (default curriculum)
- [ ] Test alternative stage splits
- [ ] Implement weighted buffer sampling
- [ ] Push best config row to RESULTS.md
