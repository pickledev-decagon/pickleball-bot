## Context

We are building a pickleball RL competition bot. The project is at:
/Users/keven/Documents/degree_study_material/y3s3/Deep_RL/pickleball_bot

The Git repo is at: https://github.com/pickledev-decagon/pickleball-bot

We have 5 team members (Member 1 to Member 5), each assigned to a different area:
- M1: Model Architecture (two_tower_cnn.py, env/wrapper.py)
- M2: Reward Functions (reward_shaping.py, experiment_configs.py)
- M3: Observation (preprocessing.py)
- M4: Hyperparameters (sweep_configs, grid_search.py)
- M5: Training Strategy (train_selfplay_v2.py, selfplay_callback.py, buffer.py, config.py)

B1 smoke test (10k steps) has PASSED — pipeline validated.

## The Problem

I reviewed the progress tracking files for alignment and found 8 issues. My original proposal was to restructure the entire progress/ folder with nested subdirectories, but the user said that was overcomplicated. Keep the flat structure, just fix the content.

Additionally, the user wants **dependency tracking** added to each member's progress file. Each member should know:
- **Depends on**: what they need from others before they can proceed
- **Depended by**: who is waiting on their results

This creates a clear map of who needs what from whom.

## The Fix Plan

Keep the flat `progress/` directory. Apply these 8 fixes plus dependency tracking:

| # | Fix | File(s) |
|---|-----|---------|
| 1 | Clarify `env/wrapper.py` ownership split — M1 owns preprocessing pipeline (frame conversion, resize, grayscale), M3 owns ball detection + heatmap rendering. Both coordinate changes via PR. | CONTRIBUTING.md (team roles table), all 5 member files |
| 2 | Add curriculum warning: `run_experiment.py` does NOT use the 3-stage curriculum (no-op → square → self-play). Experiments run via it train against Unity's default opponent. Results may not transfer to final training. Also fix M4's sweep configs where `frame_stack=4` should be `16`. | M4's member file, `sweep_configs/phase3_default.json`, `phase3_focused.json` |
| 3 | Add shared "done" criteria block to all member files: 50k steps minimum per variant, evaluate both left+right (50 matches each), win rate > 85% vs square = baseline met. | All 5 member files |
| 4 | Document M5's dependency: Stage 3 self-play needs checkpoints from other members. M5 can start with curriculum stage split experiments (Stage 1+2 only) while waiting, but full testing requires M1-M4 checkpoints. | M5's member file |
| 5 | Update status text from "⏳ Pending — waiting for B1" → "🟢 Ready to start". | All 5 member files |
| 6 | Fix M4's sweep configs: change `frame_stack: 4` to `frame_stack: 16` in both phase3_default.json and phase3_focused.json. | `sweep_configs/phase3_default.json`, `phase3_focused.json` |
| 7 | Fix M4's experiment log columns to match other members: use `Win Rate (right)`, `Win Rate (left)`, `Best Opponent` instead of `Best Win Rate`, `Key Finding`. | M4's member file |
| 8 | Standardize baseline config text across all members: "Baseline: heatmap_obs, stack_16, 50k steps". Keep consistency. | All 5 member files |
| 9 | **Add Dependency Tracking section to all 5 member files** with two tables: "Depends On" (what I need from others) and "Depended By" (who needs my results). See dependency map below. | All 5 member files |

## Dependency Map

```
M3: Observation                    M1: Architecture
  ── baseline (heatmap_obs) ──┬──▶   ── depends on M3's baseline ──▶
                              │
M2: Reward                     │     M4: Hyperparameters
  ── depends on M3's baseline ─┤──▶   ── depends on M3's baseline ──▶
                              │
                              │     M5: Training Strategy
                              │       ── depends on M1-M4 checkpoints ──▶
                              │       ── depends on M3's baseline ──────▶
```

The dependency tracking section should look like:

```markdown
## Dependency Tracking

### Depends On (what I need from others)

| What I Need | From Who | Why | Status |
|------------|----------|-----|--------|
| Stable heatmap_obs baseline | Member 3 | All my experiments use this as base config | ⏳ Not yet |
| — | — | — | — |

### Depended By (who is waiting on my results)

| Who Needs Me | What They Need | Why | Status |
|-------------|---------------|-----|--------|
| Member 5 | Trained checkpoint for self-play buffer | Stage 3 needs past checkpoints | ⏳ Not yet |
| — | — | — | — |
```

## Instructions

Apply all 9 fixes. Commit each fix as a separate commit with descriptive messages referencing the issue number (e.g. "fix #1: clarify env/wrapper.py ownership split", "fix #9: add dependency tracking to all member files"). Do NOT push — I'll tell you when to push.
