# Documentation Updates

## Files to Update

| File | What Changes |
|------|-------------|
| `EXPERIMENTS.md` | Add section 6 (TwoTower CNN), update section 2 with architecture variants, add frame stack depth to section 1.4, add LSTM note |
| `RESULTS.md` | Phase 3 comparison table |
| `PROJECT_GUIDE.md` | No changes needed |

---

## 1. EXPERIMENTS.md — Section 2 (Architecture Variants)

Replace empty placeholder with:

### 2.1 TwoTower CNN (Default in Phase 3)

| | |
|---|---|
| **What** | Separate CNN encoders for policy and value |
| **Why** | Prevents value loss corrupting policy features (Cobbe+ 2019) |
| **Code** | `agent/two_tower_cnn.py` — class `TwoTowerCnnPolicy` |
| **Config** | `PPO(TwoTowerCnnPolicy, ...)` |
| **Risk** | 2× CNN params, slightly slower |
| **Hypothesis** | Higher ep_rew_mean ceiling at 50k+ steps |

### 2.2 Spatial Softmax CNN (Alternative)

| | |
|---|---|
| **What** | Replace GAP with spatial softmax weighted positions |
| **Why** | CNN learns ball (x,y) without BallTracker |
| **Code** | `agent/spatial_softmax_cnn.py` (optional) |
| **Risk** | Unproven in this domain |

Quick config table row:

| Config | Quick Why |
|--------|-----------|
| `two_tower` | Separate CNN heads for policy vs value |

---

## 2. EXPERIMENTS.md — Section 1.4 (Frame Stack)

Add:

| Config | Value | Expected |
|--------|-------|----------|
| `stack_2` | 2 | Faster, less temporal info |
| `baseline` | 4 | Current default |
| `stack_8` | 8 | More context, ~2× channels |
| `stack_16` | 16 | Max context, ~4× channels |

---

## 3. EXPERIMENTS.md — New Section 6 (Model Architecture)

### 6. TwoTower CNN Training

| | |
|---|---|
| **Config name** | `two_tower` |
| **Code** | `agent/two_tower_cnn.py` |
| **Sweep** | `scripts/grid_search.py` |
| **Run cmd** | `python run_experiment.py --config two_tower --steps 50000 --envs 4` |

---

## 4. EXPERIMENTS.md — LSTM Note

After section 2:

> **LSTM / RecurrentPPO:** Evaluated in `docs/architecture_decisions.md`.
> Decision: skip Phase 3. Frame stack sufficient. Revisit if ball occlusion
> is a problem.

---

## 5. RESULTS.md — Phase 3 Table

### Phase 3 — Model Architecture

| Config | Steps | FPS | Entropy | Value Loss | Ep Rew | Params | Notes |
|--------|-------|-----|---------|------------|--------|--------|-------|
| baseline (shared CNN) | 10k | — | — | — | — | ~0.5M | |
| two_tower | 10k | — | — | — | — | ~1.0M | 2× CNN |

### Phase 3 — Grid Search Best

| Trial | ent_coef | lr | n_steps | batch | clip | Epochs | FPS | Ep Rew |
|-------|----------|----|---------|-------|------|--------|-----|--------|
| best | ... | ... | ... | ... | ... | ... | ... | ... |

---

## Implementation Notes

- Phase 3 code files must exist before doc updates are committed
- Exception: `stack_8`/`stack_16` in section 1.4 can be updated after Member F
  adds them to `experiment_configs.py`
