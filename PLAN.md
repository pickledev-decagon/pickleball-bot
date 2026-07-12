# Plan: Validate Pipeline → Component-by-Component → Full Prototype

> **Status:** Ready to start. All 4 critical bugs fixed (commit 131b986).  
> **Current branch:** `team/hyperparams`  
> **Initial model:** Fresh random PPO CnnPolicy (no pretrained weights)  
> **Saved:** 2026-06-26

## Context

After deep investigation, we found 4 critical bugs that made all prior training useless:
1. Double normalization (`/ 255.0` on already-[0,1] frames → near-black)
2. 1D hit detection (only X-axis, missing vertical dinks/lobs)
3. Rotation penalty = 0.0 in all experiment runs
4. Square opponent itself spinning (normalizing spin behavior)

All 4 are now fixed. All previous checkpoints are garbage. We start fresh.

**Initial model for this plan:** Fresh random-initialized PPO with SB3 `CnnPolicy`. No pretrained weights. `PPO("CnnPolicy", vec_env)` creates random CNN + value network at each experiment start.

## Team Responsibility Map

| Person | Component | How to validate |
|--------|-----------|-----------------|
| Keven | PPO hyperparams, experiment harness | Compare lr/batch/entropy configs |
| Yenly | Reward shaping | Compare hit/aggression/placement/rotation weights |
| Siao Xuan | Observations | Confirm best heatmap config (not full re-sweep) |
| Amber | TwoTowerCNN architecture | Compare CnnPolicy vs TwoTowerCNN |
| Hui Xuan | Curriculum + self-play | Compare no-curriculum vs curriculum |

## Rules for All Experiments

1. **Every run produces**: JSON with training metrics + spin rate (from auto-measurement in run_experiment.py)
2. **Fixed cost per step**: n_steps=2048//num_envs, batch_size=128, n_epochs=4
3. **All runs use SEED=42** (single seed for speed; multi-seed only for promising finalists)
4. **Screening eval**: right-side only vs no_op, 10 matches (~10 min). Fast enough to decide go/no-go.
5. **Full eval**: both sides × 3 opponents, 20 matches each (~2 hours). Only on the best checkpoint from each phase.
6. **Compare direction not just destination**: is win rate climbing? Is entropy dropping? Not just absolute numbers.

---

## Phase 0: Pipeline Validation (no training, ~5 min)

**Goal:** Confirm the 4 bug fixes work and frames are bright.

### 0a. Run `test_env.py`
```
python test_env.py
```
**Pass criteria:**
- `obs max >= 0.95` (now that /255.0 is removed)
- `obs.shape == (1, 4, 84, 168)` float32
- `action.shape == (3,)` values in [0,2]
- No crash

### 0b. Verify heatmap sync
```
python scripts/validate_export.py
```
**Pass:** 53/53 tests. Confirms wrapper.py and competition_export.py produce identical output.

### 0c. Quick frame capture inspection
Run one env, capture a few frames, verify they look like a pickleball court (not black).

**Gate:** All 3 pass → proceed to Phase 1.

---

## Phase 1: Sparse Only — Does PPO Learn At All?

**Goal:** Prove PPO can improve from a random policy toward making contact, given only win/loss signal and proper visual input. This is the "hello world" — simplest possible config.

**Initial model:** Random PPO CnnPolicy (no pretrained weights).

**Config:** `sparse_only` — grayscale, 4-stack, no heatmap, no reward shaping, no curriculum.

### 1a. Smoke test (5k steps, ~2 min)
```
python run_experiment.py --config sparse_only --steps 5000 --envs 2 --opponent no_op
```

### 1b. 50k run (~25 min)
```
python run_experiment.py --config sparse_only --steps 50000 --envs 2 --opponent no_op
```

**Pass criteria (direction, not absolute):**
| Metric | 5k → 50k trend | What it means |
|--------|----------------|---------------|
| Win rate vs no_op | Increasing | Learning to score, not random |
| Spin rate | <20% at 50k | Not completely random rotation |
| Entropy | Decreasing | Policy confidence growing |
| Explained variance | > -0.01 at 50k | Value function at least slightly useful |

These are low bars. We just need to see improvement from 5k to 50k — proof that PPO is receiving a useful signal.

**What if we fail?** If entropy stays flat and win rate stays at 0%, the Unity sparse reward might not be wired correctly, or 50k is simply too few steps for sparse-only PPO. Fallback: skip to Phase 3 (add proximity reward) and retry — proximity gives dense signal every step.

**Gate:** Clear improvement trend → pipeline confirmed working → proceed to Phase 2.

---

## Phase 2: Observations — Siao Xuan's Domain

**Goal:** Confirm the best observation config using our fixed pipeline. NOT redoing Siao Xuan's full sweep — just A/B testing the two extremes: bare grayscale vs best heatmap config.

**Team:** Siao Xuan's component validated.

**Initial model:** Random PPO CnnPolicy (each run starts fresh).

**Baseline:** Phase 1's best sparse-only checkpoint.

### 2a. Frame stack: 4 vs 16 (A/B test)
Train `stack_16` config (= sparse_only + frame_stack=16) at 10k steps, compare to Phase 1's frame_stack=4.

**Pass:** 16-stack improves win rate by >5% OR entropy drops faster.

### 2b. Heatmap: on vs off (A/B test)
Train `heatmap_obs` (heatmap + 4-stack, sigma=5, weight=0.7 — Siao Xuan's confirmed best) at 10k steps, compare to no-heatmap.

**Pass:** Heatmap improves win rate by >5% OR rally length increases.

### 2c. Quick combo test (only if both 2a and 2b pass)
Train `heatmap_w07_s5` (heatmap + sigma=5 + weight=0.7 + stack=16) at 10k steps.
This is the Siao Xuan "best confirmed combo" — validate it in our fixed pipeline.

**Pass:** Combo matches or beats the individual improvements from 2a and 2b.

**Gate:** Best config from this phase becomes the **canonical observation config**.

---

## Phase 3: Rewards — Yenly's Domain

**Goal:** Add reward components one at a time and measure marginal benefit. Each component earns its keep or gets dropped.

**Team:** Yenly's component validated.

**Initial model:** Random PPO CnnPolicy (each run starts fresh).

**Baseline:** Phase 2's best observation config + sparse reward only.

**All runs at 20k steps** (~10 min each) for quick comparison.

### 3a. Proximity reward (on/off)
Add `proximity_weight=1.0`. Compare to baseline.

**Pass:** Win rate improves >5% over baseline at same step count.

### 3b. Hit bonus (0.5, 1.0)
Test `hit_bonus=0.5` and `hit_bonus=1.0` (on top of proximity). Compare at 20k steps each.

**Pass:** Rally length increases >2 steps over 3a, win rate doesn't regress.

### 3c. Rotation penalty validation
Compare `rotation_penalty=0.01` vs `rotation_penalty=0.0` at 10k steps (on top of proximity + hit).

**Pass:** Spin rate <5% with penalty, win rate not >5% worse than without.

### 3d. Aggression bonus (on/off, optional)
If 3b passes, test aggression_bonus=0.01.

**Pass:** Point differential improves vs no_op.

**Gate:** Best reward config becomes the **canonical reward config**.

---

## Phase 4: Validate `train_selfplay_v2.py` Path

**Goal:** Confirm the main training script works with the best config from Phase 3. `train_selfplay_v2.py` has its own hardcoded observation + reward — we need to verify this code path produces the same results.

**Team:** Cross-validation of Hui Xuan's training infrastructure.

**Action:** Update `train_selfplay_v2.py` to use the best config from Phase 3 (or verify its hardcoded values match). Run one 10k-step smoke test to confirm:
- Same observation pipeline code path works
- CurriculumCallback doesn't crash
- CheckpointCallback saves metadata correctly

**Gate:** train_selfplay_v2.py smoke test passes → both training paths validated.

---

## Phase 5: Opponent Curriculum — Hui Xuan's Domain

**Goal:** Prove curriculum training (Stage 1 no_op → Stage 2 square) beats flat training.

**Team:** Hui Xuan's component validated.

**Initial model:** Random PPO CnnPolicy (each run starts fresh).

**Baseline:** Phase 3's best config (best obs + best reward), no curriculum.

### 5a. Opponent type impact (no curriculum)
Train 3 separate runs at 20k each: vs no_op, vs random, vs square.
All with Phase 3's best obs + reward, curriculum=False.
Measure: does training vs square transfer to beating no_op?

**Pass:** Square-trained model reaches >30% win rate vs square. (Square is hard — this is a moderate bar.)

### 5b. Curriculum on/off (50k validation)
Compare `curriculum=True` vs `curriculum=False` at 50k steps (~25 min each).
Both use Phase 3's config.
Screening eval (right-only vs no_op, 10 matches) to compare.

**Pass criteria:**
- Curriculum model beats non-curriculum by >5% win rate vs square
- No learning cliff at stage transition (explained_variance doesn't crash to negative)

### 5c. Full curriculum (100k+)
Take the winner from 5b and scale to 100-200k steps with 8 envs and full eval.
This is the candidate competition model.

**Pass:** Beats no_op >80%, beats random >60%, beats square >40%.

**Gate:** Curriculum model is the **canonical training strategy**.

---

## Phase 6: Model Architecture — Amber's Domain

**Goal:** Test TwoTowerCNN vs default CnnPolicy.

**Prerequisite:** Wire TwoTowerCNN into `run_experiment.py` (currently only accessible via factory functions, not used by any training path).

**Team:** Amber's component validated.

### 6a. Architecture comparison
Train TwoTowerCNN vs CnnPolicy at 100k steps with Phase 3 config + Phase 5 curriculum.
Full eval on both.

**Pass:** TwoTowerCNN either:
- Beats CnnPolicy by >5% win rate, OR
- Shows more stable training (lower value_loss variance), OR
- Equivalent → CnnPolicy wins (simpler is better)

**Gate:** Best architecture is the **final model**.

---

## Phase 7: Hyperparameter Tuning — Keven's Domain

**Goal:** Fine-tune PPO hyperparams with best arch + obs + reward + curriculum.

**Initial model:** Random PPO (best architecture from Phase 6). No prior weights.

**Baseline:** Phase 6's best model with default PPO params.

### 7a. Learning rate sweep
Compare lr ∈ {1e-4, 2.5e-4, 5e-4} at 50k each.

### 7b. Entropy sweep
Compare ent_coef ∈ {0.001, 0.01, 0.02} at 50k each.

### 7c. Batch size sweep
Compare batch_size ∈ {64, 128, 256} at 50k each.

### 7d. Best combo
Best lr + best ent + best batch from 7a-7c, run at 200k with full eval.

**Gate:** This is the **competition entry model**.

---

## Summary

```
Phase 0: Pipeline validation ── 5 min ─── FAIL → fix bugs, retry
                                        └─ PASS ↓
Phase 1: Sparse only ───────── 30 min ── Check: is PPO improving at all?
                                        └─ PASS ↓
Phase 2: Best observation ──── 20 min ─── A/B: no-heatmap vs heatmap
                                        └─ Canonical obs config ↓
Phase 3: Best reward ───────── 60 min ── Add proximity → hit → rotation, measure marginal gain
                                        └─ Canonical reward config ↓
Phase 4: Validate selfplay ─── 10 min ─── Smoke test train_selfplay_v2.py
                                        └─ Both paths confirmed ↓
Phase 5: Curriculum ────────── 100 min ── Compare curriculum on/off, scale winner
                                        └─ Canonical training strategy ↓
Phase 6: TwoTowerCNN ───────── 60 min ─── Wire + compare vs CnnPolicy
                                        └─ Final architecture ↓
Phase 7: Hyperparam tuning ─── 3 hours ── Sweep lr/ent/batch, combine best
                                        └─ Competition entry model
```

## Evaluation Strategy

| When | What | Time |
|------|------|------|
| Every experiment (auto) | Spin rate measurement | ~1 min |
| Screening (within-phase decisions) | Right-side vs no_op, 10 matches | ~10 min |
| Phase best (phase gate) | Both sides × 3 opponents, 20 matches | ~2 hours |
| Final model | Both sides × 4 opponents, 50 matches | ~4 hours |

## Key Files

| File | Purpose |
|------|---------|
| `test_env.py` | Pipeline end-to-end validation |
| `run_experiment.py` | Experiment runner (all phases 1-7 experiments) |
| `evaluate_checkpoint.py` | Load checkpoint, measure win rates |
| `compare_results.py` | Side-by-side experiment comparison |
| `agent/experiment_configs.py` | All experiment configs |
| `agent/reward_shaping.py` | Reward components |
| `agent/train_selfplay_v2.py` | Curriculum training script |
| `agent/two_tower_cnn.py` | TwoTowerCNN (needs wiring for Phase 6) |
| `scripts/validate_export.py` | Heatmap sync verification |

## Refinement Report (from plan review)

Changes made based on critical review:
1. **Phase 1 criteria changed**: absolute win rate → direction of improvement (win rate climbing, entropy dropping, spin <20%)
2. **Evaluation scoped**: screening eval (right-only, 10 matches) for quick decisions; full eval only on phase-best checkpoints
3. **Phase 5 curriculum**: first compare at 50k (not 100k), then scale winner
4. **Phase 2 scope reduced**: A/B test no-heatmap vs best-heatmap only (not full re-sweep of all sigma/weight combos)
5. **Phase 6 prep added**: TwoTowerCNN must be wired into run_experiment.py before experiments
6. **Phase 4 added**: validate train_selfplay_v2.py code path (not just run_experiment.py)
7. **Phase 1 fallback added**: if sparse-only fails, skip to proximity reward — dense signal may be needed
