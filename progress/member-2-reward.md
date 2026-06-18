# Area: Reward Functions — Yenly

## Assigned Files

| File | Role |
|------|------|
| `agent/reward_shaping.py` | proximity, hit detection, placement, aggression rewards |
| `agent/experiment_configs.py` | Named experiment presets (reward configs) |

> **`env/wrapper.py` split awareness:** Amber owns the preprocessing pipeline (frame conversion, resize, grayscale); Siao Xuan owns ball detection + heatmap rendering. If your reward config needs new observation features, coordinate via PR with the relevant owner.

---

## Current Status

**🟡 Baseline preliminary 10k trial completed; reward variants pending**

---

## Experiment Log

| # | Date | Config | Steps | Win Rate (right) | Win Rate (left) | Best Opponent | Drive Link | Notes |
|---|------|--------|-------|-----------------|-----------------|--------------|------------|-------|
| 1 | 2026-06-13 | baseline | 10,000 | N/A | N/A | N/A | N/A | Preliminary trial. Stable PPO metrics. Result: `results/baseline_20260613_163125.json` |
| 2 | 2026-06-13 | aggression_bonus | 10,000 | N/A | N/A | N/A | N/A | Preliminary 10k trial. Result file: `results/aggression_bonus_20260613_180422.json`. Steps/sec=42.7, entropy=-3.239, approx_kl=0.00481, value_loss=0.5708. Stable PPO update, but value loss is much higher than baseline, so this reward needs comparison with other variants before selecting. |
| 3 | 2026-06-13 | hit_bonus_high | 10,000 | N/A | N/A | N/A | N/A | Preliminary 10k trial. Result file: `results/hit_bonus_high_20260613_183329.json`. Steps/sec=42.1, entropy=-3.294, approx_kl=0.00316, value_loss=0.0459. Stable PPO update; value loss is close to baseline and much lower than aggression_bonus. |
| 4 | 2026-06-13 | placement_bonus | 10,000 | N/A | N/A | N/A | N/A | Preliminary 10k trial. Result file: `results/placement_bonus_20260613_184520.json`. Steps/sec=44.9, entropy=-3.269, approx_kl=0.00752, value_loss=0.1017. Stable PPO update, fastest trial so far, but KL and value loss are higher than baseline and hit_bonus_high. |
| 5 | 2026-06-13 | hit_bonus_075 | 10,000 | N/A | N/A | N/A | N/A | Preliminary 10k trial. Result file: `results/hit_bonus_075_20260613_185748.json`. Steps/sec=43.9, entropy=-3.295, approx_kl=0.00506, value_loss=0.0473. Stable PPO update; performance is very close to hit_bonus_high, but with slightly higher KL and value loss. |
| 6 | 2026-06-13 | hit_bonus_125 | 10,000 | N/A | N/A | N/A | N/A | Preliminary 10k trial. Result file: `results/hit_bonus_125_20260613_190814.json`. Steps/sec=47.0, entropy=-3.275, approx_kl=0.00721, value_loss=0.0854. Stable PPO update and fastest hit-bonus trial so far, but KL and value loss are higher than hit_bonus_high and hit_bonus_075, suggesting 1.25 may be stronger but less stable. |
| 7 | 2026-06-13 | aggression_bonus_005 | 10,000 | N/A | N/A | N/A | N/A | Preliminary 10k trial. Result file: `results/aggression_bonus_005_20260613_192230.json`. Steps/sec=39.9, entropy=-3.272, approx_kl=0.00363, value_loss=0.1760. Stable PPO update; value loss is much lower than aggression_bonus=0.10 but still higher than baseline and hit_bonus configs. |
| 8 | 2026-06-13 | aggression_bonus_001 | 10,000 | N/A | N/A | N/A | N/A | Preliminary 10k trial. Result file: `results/aggression_bonus_001_20260613_230143.json`. Steps/sec=45.3, entropy=-3.295, approx_kl=0.00045, value_loss=0.0366. Very stable PPO update; value loss is close to baseline and much lower than aggression_bonus_005 and aggression_bonus=0.10. |
| 9 | 2026-06-13 | aggression_bonus_002 | 10,000 | N/A | N/A | N/A | N/A | Preliminary 10k trial. Result file: `results/aggression_bonus_002_20260613_231733.json`. Steps/sec=45.9, entropy=-3.293, approx_kl=0.00231, value_loss=0.0539. Stable PPO update, but value loss and KL are higher than aggression_bonus_001. |
| 10 | 2026-06-13 | aggression_bonus_0015 | 10,000 | N/A | N/A | N/A | N/A | Preliminary 10k trial. Result file: `results/aggression_bonus_0015_20260613_232725.json`. Steps/sec=45.6, entropy=-3.286, approx_kl=0.00202, value_loss=0.0412. Stable PPO update; performance is between aggression_bonus_001 and aggression_bonus_002, but 0.01 still has the lowest KL and value loss. |
| 11 | 2026-06-13 | placement_bonus_010 | 10,000 | N/A | N/A | N/A | N/A | Preliminary 10k trial. Result file: `results/placement_bonus_010_20260613_233948.json`. Steps/sec=45.5, entropy=-3.285, approx_kl=0.00937, value_loss=0.0680. Stable PPO update, but approx_kl is close to 0.01; value loss is lower than placement_bonus=0.30. |
| 12 | 2026-06-13 | placement_bonus_020 | 10,000 | N/A | N/A | N/A | N/A | Preliminary 10k trial. Result file: `results/placement_bonus_020_20260613_234705.json`. Steps/sec=44.5, entropy=-3.293, approx_kl=0.00547, value_loss=0.1731. Stable PPO update, but value loss is higher than placement_bonus_010 and placement_bonus=0.30, so 0.20 is not the best placement candidate. |
| 13 | 2026-06-13 | placement_bonus_005 | 10,000 | N/A | N/A | N/A | N/A | Preliminary 10k trial. Result file: `results/placement_bonus_005_20260613_235412.json`. Steps/sec=45.4, entropy=-3.262, approx_kl=0.00679, value_loss=0.0426. Stable PPO update; value loss is close to baseline and much lower than placement_bonus_010, 0.20, and 0.30. |
| 14 | 2026-06-14 | placement_bonus_0025 | 10,000 | N/A | N/A | N/A | N/A | Preliminary 10k trial. Result file: `results/placement_bonus_0025_20260614_000348.json`. Steps/sec=44.2, entropy=-3.295, approx_kl=0.000086, value_loss=0.0285. Very stable PPO update; value loss and KL are lower than baseline, while still adding a small shot-quality reward. |
| 15 | 2026-06-14 | hit_bonus_09 | 10,000 | N/A | N/A | N/A | N/A | Preliminary 10k trial. Result file: `results/hit_bonus_09_20260614_001851.json`. Steps/sec=44.9, entropy=-3.274, approx_kl=0.00228, value_loss=0.0676. Stable PPO update, but value loss is higher than hit_bonus_high=1.0, so 0.9 is not clearly better. |
| 16 | 2026-06-14 | hit_bonus_11 | 10,000 | N/A | N/A | N/A | N/A | Preliminary 10k trial. Result file: `results/hit_bonus_11_20260614_002445.json`. Steps/sec=44.7, entropy=-3.294, approx_kl=0.00559, value_loss=0.0629. Stable PPO update, but both KL and value loss are higher than hit_bonus_high=1.0, so 1.1 is not better than 1.0. |
| 17 | 2026-06-14 | hit_aggression_placement_combo | 10,000 | N/A | N/A | N/A | N/A | Preliminary 10k combination trial using current baseline observation. Result file: `results/hit_aggression_placement_combo_20260614_003733.json`. Combines hit_bonus=1.0, aggression_bonus=0.01, and placement_bonus=0.025. Steps/sec=44.3, entropy=-3.281, approx_kl=0.00274, value_loss=0.0657. Stable PPO update, but value loss is higher than the best individual reward settings. Final combo should be rerun after the observation setup is confirmed. |

---
## Reward Trial Comparison

### 1. Hit Bonus Comparison

| Config | Hit Bonus Value | Steps | Envs | Steps/sec | Entropy | Approx KL | Value Loss | Stability | Interpretation | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| baseline | 0.50 | 10,000 | 4 | 40.1 | -3.288 | 0.00462 | 0.0337 | Stable | Baseline hit reward; used as reference | Keep as baseline |
| hit_bonus_075 | 0.75 | 10,000 | 4 | 43.9 | -3.295 | 0.00506 | 0.0473 | Stable | Moderate hit reward between baseline and high hit bonus | Stable, but not better than 1.0 |
| hit_bonus_09 | 0.90 | 10,000 | 4 | 44.9 | -3.274 | 0.00228 | 0.0676 | Stable | Slightly lower than 1.0; tests whether milder hit reward improves stability | Stable, but value loss is higher than 1.0 |
| #hit_bonus_high | 1.00 | 10,000 | 4 | 42.1 | -3.294 | 0.00316 | 0.0459 | Stable | Strong hit reward; encourages paddle contact | Best hit-bonus candidate so far |
| hit_bonus_11 | 1.10 | 10,000 | 4 | 44.7 | -3.294 | 0.00559 | 0.0629 | Stable | Slightly higher than 1.0; tests whether stronger hit reward improves learning | Stable, but weaker than 1.0 |
| hit_bonus_125 | 1.25 | 10,000 | 4 | 47.0 | -3.275 | 0.00721 | 0.0854 | Stable, but less stable | Stronger hit reward than 1.0; may increase reward pressure | Keep for comparison, but 1.0 looks safer |

**Hit bonus interpretation:**  
Among the tested hit-bonus values, `hit_bonus_high` with `hit_bonus=1.0` currently looks like the strongest candidate. It has the lowest `approx_kl` and a low `value_loss`, while still keeping healthy entropy. Increasing the value to `1.25` did not improve stability, so testing `1.5` is not necessary at this stage.

### 2. Aggression Bonus Comparison

| Config | Aggression Bonus Value | Steps | Envs | Steps/sec | Entropy | Approx KL | Value Loss | Stability | Interpretation | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| baseline | 0.00 | 10,000 | 4 | 40.1 | -3.288 | 0.00462 | 0.0337 | Stable | No aggression bonus; used as reference | Keep as baseline |
| #aggression_bonus_001 | 0.01 | 10,000 | 4 | 45.3 | -3.295 | 0.00045 | 0.0366 | Very stable | Very mild attacking reward; encourages opponent-side play without strong reward pressure | Best aggression candidate so far |
| aggression_bonus_0015 | 0.015 | 10,000 | 4 | 45.6 | -3.286 | 0.00202 | 0.0412 | Stable | Fine-tuned aggression reward between 0.01 and 0.02 | Stable, but slightly weaker than 0.01 |
| aggression_bonus_002 | 0.02 | 10,000 | 4 | 45.9 | -3.293 | 0.00231 | 0.0539 | Stable | Mild attacking reward; still stable, but stronger reward pressure than 0.01 | Keep for comparison, but 0.01 is better |
| aggression_bonus_005 | 0.05 | 10,000 | 4 | 39.9 | -3.272 | 0.00363 | 0.1760 | Stable, but critic loss higher than baseline | Milder attacking reward, but still increases critic loss clearly | Less suitable than 0.01/0.015/0.02 |
| aggression_bonus | 0.10 | 10,000 | 4 | 42.7 | -3.239 | 0.00481 | 0.5708 | PPO update stable, but critic loss high | Stronger attacking reward; encourages attacking play | Too strong for now |

**Aggression interpretation:**  
The `aggression_bonus` run remained stable in terms of `approx_kl`, but its `value_loss` increased sharply compared with the baseline. This suggests the critic found the reward signal harder to estimate. The aggression bonus should not be selected yet without testing a smaller value, such as `aggression_bonus=0.05`, or combining it carefully with a more stable reward component.

### 3. Placement Bonus Comparison

| Config | Placement Bonus Value | Steps | Envs | Steps/sec | Entropy | Approx KL | Value Loss | Stability | Interpretation | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| baseline | 0.00 | 10,000 | 4 | 40.1 | -3.288 | 0.00462 | 0.0337 | Stable | No placement bonus; used as reference | Keep as baseline |
| #placement_bonus_0025 | 0.025 | 10,000 | 4 | 44.2 | -3.295 | 0.000086 | 0.0285 | Very stable | Very small shot-quality reward; adds placement guidance with minimal reward pressure | Best placement candidate so far |
| placement_bonus_005 | 0.05 | 10,000 | 4 | 45.4 | -3.262 | 0.00679 | 0.0426 | Stable | Very mild shot-quality reward; useful but slightly more reward pressure than 0.025 | Second-best placement candidate |
| placement_bonus_010 | 0.10 | 10,000 | 4 | 45.5 | -3.285 | 0.00937 | 0.0680 | Stable, but KL close to 0.01 | Mild shot-quality reward for corner/far-back placement | Keep for comparison, but weaker than 0.025/0.05 |
| placement_bonus_020 | 0.20 | 10,000 | 4 | 44.5 | -3.293 | 0.00547 | 0.1731 | Stable, but critic loss high | Moderate shot-quality reward; KL acceptable but value loss increased clearly | Not recommended |
| placement_bonus | 0.30 | 10,000 | 4 | 44.9 | -3.269 | 0.00752 | 0.1017 | Stable, but critic loss higher than 0.025/0.05 | Stronger shot-quality reward; may add more reward pressure | Not the best placement setting |

**Placement interpretation:**  
The `placement_bonus` run was still stable because `approx_kl` stayed below `0.01`, but both `approx_kl` and `value_loss` were higher than the baseline and `hit_bonus_high`. This means placement reward may be useful, but it should be tested as a combination with the more stable `hit_bonus=1.0` before selection.

### 4. Combined Reward Comparison

| Config | Hit Bonus | Aggression Bonus | Placement Bonus | Steps | Envs | Steps/sec | Entropy | Approx KL | Value Loss | Stability | Interpretation | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| baseline | 0.50 | 0.00 | 0.00 | 10,000 | 4 | 40.1 | -3.288 | 0.00462 | 0.0337 | Stable | Default reward setting used as reference | Keep as baseline |
| hit_bonus_high | 1.00 | 0.00 | 0.00 | 10,000 | 4 | 42.1 | -3.294 | 0.00316 | 0.0459 | Stable | Best individual hit-bonus setting | Strong individual candidate |
| aggression_bonus_001 | 0.50 | 0.01 | 0.00 | 10,000 | 4 | 45.3 | -3.295 | 0.00045 | 0.0366 | Very stable | Best individual aggression setting | Strong individual candidate |
| placement_bonus_0025 | 0.50 | 0.00 | 0.025 | 10,000 | 4 | 44.2 | -3.295 | 0.000086 | 0.0285 | Very stable | Best individual placement setting | Strongest individual training metric |
| hit_aggression_placement_combo | 1.00 | 0.01 | 0.025 | 10,000 | 4 | 44.3 | -3.281 | 0.00274 | 0.0657 | Stable | Combines the best hit, aggression, and placement values | Preliminary only; rerun after final observation setup |

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

- [x] **Baseline** — proximity_weight=1.0, hit_bonus=0.5, aggression=0.0, placement=0.0
- [x] **Aggression bonus** — +0.1 per frame ball on opponent side
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
