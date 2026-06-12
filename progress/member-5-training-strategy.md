# Area: Training Strategy — Member 5

## Assigned Files

| File | Role |
|------|------|
| `agent/train_selfplay_v2.py` | Main training script with curriculum |
| `agent/selfplay_callback.py` | Mid-training checkpoint → buffer logic |
| `agent/buffer.py` | OpponentBuffer (sampling, eviction) |
| `config.py` | Central config (training constants) |

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

- [ ] Run default curriculum baseline (300k steps)
- [ ] Test alternative stage splits
- [ ] Implement weighted buffer sampling
- [ ] Push best config row to RESULTS.md
