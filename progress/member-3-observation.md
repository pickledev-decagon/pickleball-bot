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

## What's Already Known (Phase 2 — Old 4-Stack Pipeline)

These were tested before the overhaul. Results are directional — re-validate with current pipeline:

| Config | Finding | Verdict |
|--------|---------|---------|
| **heatmap_obs** ⭐ | Explained variance **0.329** (36× better than baseline) | ✅ **Winner — use as baseline** |
| rgb_obs | 25% slower FPS, negligible EV improvement | ❌ Skip — not worth re-testing |
| frame_diff | Near-zero frames, slow | ❌ Skip — not worth re-testing |
| stack_2 | Too little temporal info | ❌ Skip — current stack_16 is better |

**Your job is not to re-test everything.** Focus on validating the heatmap baseline with the current pipeline, then explore what wasn't possible before (stack_32, heatmap+stack_32, SmoothBallTracker tuning).

---

## Variants to Test

| Priority | Variant | Config | Why | Est. Time |
|----------|---------|--------|-----|-----------|
| **P0** 🔴 | **Heatmap baseline (stack_16)** | `--config heatmap_obs` | **Everyone else is blocked waiting for this.** Run 50k steps, evaluate both sides. | ~2-3 hrs |
| **P1** 🟡 | **Stack_32** | `--config stack_32` | 800ms context vs 400ms. Was never tested — might help wall-bounce prediction. | ~2-3 hrs |
| **P1** 🟡 | **Heatmap + stack_32** | Code change needed | Combine best obs (heatmap) with deeper context. The top candidate. | ~3-4 hrs |
| **P2** ⚪ | **No heatmap (plain grayscale)** | `--config baseline` | Control comparison — is heatmap still winning with 16-stack? | ~2 hrs |
| **P2** ⚪ | **SmoothBallTracker tuning** | Already wired | Fine-tune occlusion handling if ball tracking has gaps | ~1 hr |

---

## What to Run First

```bash
conda activate dpickleball

# 🔴 P0 — Run this immediately. Everyone needs it:
python run_experiment.py --config heatmap_obs --steps 50000 --envs 8

# After baseline passes Done Criteria (>85% vs square, both sides):
# 🟡 P1 — Explore new variants:
python run_experiment.py --config stack_32 --steps 50000 --envs 8

# ⚪ P2 — Validation / fine-tuning (after P0 and P1 done):
python run_experiment.py --config baseline --steps 50000 --envs 8
```

**Critical reminder:** After each experiment, upload checkpoint to the [shared Drive folder](https://drive.google.com/drive/folders/1Mi7xp03CszctgAkf6dkiUdYPrH6Z5ibz), paste the link in the experiment log below, and update `RESULTS.md`.

**Do NOT modify** `agent/two_tower_cnn.py` (Amber's), `agent/reward_shaping.py` (Yenly's), or `config.py` (Hui Xuan's).

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

- [ ] **🟢 P0:** Run heatmap baseline (heatmap_obs, stack_16, 50k steps) — everyone is waiting
- [ ] **🟡 P1:** Test stack_32 (800ms temporal context)
- [ ] **🟡 P1:** Test heatmap + stack_32 (code change needed)
- [ ] **⚪ P2:** Compare with no-heatmap grayscale baseline
- [ ] **⚪ P2:** Tune SmoothBallTracker if needed
- [ ] Push best config row to RESULTS.md
