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
| 18 | 2026-06-19 | hit_bonus_high | 10,000 | N/A | N/A | N/A | N/A | Re-tested using Siao Xuan’s updated observation setup. Result file: `results/hit_bonus_high_20260619_001402.json`. Steps/sec=39.7, entropy=-3.290, approx_kl=0.00218, value_loss=0.0793, explained_variance=0.00818. PPO update is stable because KL remains below 0.01, but value loss is higher than the previous old-observation hit_bonus_high trial. |

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
## Reward Result (merged with updated observation setup)
*previously optimized reward parameters validated under updated observation setup

### Hit Bonus:
| Config | Observation Version | Steps | Envs | Steps/sec | Entropy | Approx KL | Value Loss | Explained Variance | Stability | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| hit_bonus_high | Old/default observation | 10,000 | 4 | 42.1 | -3.294 | 0.00316 | 0.0459 | N/A | Stable | Best hit-bonus setting from earlier sensitivity test |
| hit_bonus_high | Updated observation | 10,000 | 4 | 39.7 | -3.290 | 0.00218 | 0.0793 | 0.00818 | Stable | Still stable after observation update, but critic loss increased |

### Aggression Bonus:
| Config | Observation Version | Steps | Envs | Steps/sec | Entropy | Approx KL | Value Loss | Explained Variance | Stability | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| aggression_bonus_001 | Old/default observation | 10,000 | 4 | 45.3 | -3.295 | 0.00045 | 0.0366 | N/A | Very stable | Best aggression-bonus setting from earlier sensitivity test |
| aggression_bonus_001 | Updated observation | 10,000 | 4 | 34.2 | -3.293 | 0.00127 | 0.0385 | -0.000079 | Very stable | Still stable after observation update; value loss remains close to the old-observation result, but training speed decreased |

### Placement Bonus:
| Config | Observation Version | Steps | Envs | Steps/sec | Entropy | Approx KL | Value Loss | Explained Variance | Stability | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| placement_bonus_0025 | Old/default observation | 10,000 | 4 | 44.2 | -3.295 | 0.000086 | 0.0285 | N/A | Very stable | Best placement-bonus setting from earlier sensitivity test |
| placement_bonus_0025 | Updated observation | 10,000 | 4 | 36.5 | -3.295 | 0.00175 | 0.0512 | 0.00617 | Stable | Still stable after observation update, but value loss increased and training speed decreased |

### Combine Reward:
| Config | Observation Version | Steps | Envs | Steps/sec | Entropy | Approx KL | Value Loss | Explained Variance | Stability | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| hit_aggression_placement_combo | Old/default observation | 10,000 | 4 | 44.3 | -3.281 | 0.00274 | 0.0657 | N/A | Stable | Preliminary combo using the best individual reward parameters from earlier sensitivity tests |
| hit_aggression_placement_combo | Updated observation | 10,000 | 4 | 35.8 | -3.292 | 0.00086 | 0.0486 | 0.00057 | Very stable | Combo remains stable after observation update; approx KL and value loss are both lower than the old-observation combo result, but training speed decreased |

The combined reward configuration remains stable after integrating Siao Xuan’s updated observation setup. Compared with the old/default observation combo result, the updated-observation combo recorded lower approx KL and lower value loss, suggesting a more stable PPO update and critic learning process. However, the training speed decreased from 44.3 steps/sec to 35.8 steps/sec. Overall, the updated-observation combo can still be considered a suitable final candidate for longer training.

Based on the reward sensitivity experiments, the selected reward configuration is `hit_aggression_placement_combo`, which combines `hit_bonus=1.0`, `aggression_bonus=0.01`, and `placement_bonus=0.025`. These values were first selected from the individual reward-parameter tests under the default observation setup. After rebasing with the updated observation setup, the selected values were re-tested individually and then combined. The final combined configuration remained stable, with low approximate KL divergence and moderate value loss. Therefore, this configuration is selected as the final reward-shaping candidate for longer training.

### Final Handoff-Aligned Reward Combo Training Result

| Config | Observation Version | Steps | Envs | Steps/sec | Entropy | Approx KL | Value Loss | Explained Variance | Stability | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| hit_aggression_placement_combo | Baseline observation | 50,000 | 8 | 39.4 | -3.197 | 0.00423 | 0.0522 | 0.16569 | Stable | Reward combo trained with baseline observation; useful as comparison but not final handoff-aligned setup |
| hit_aggression_placement_combo_heatmap | Heatmap + 16-frame stack | 50,000 | 8 | 37.1 | -3.216 | 0.00701 | 0.0389 | 0.76272 | Stable | Final reward combo using Siao Xuan’s updated observation setup; value loss decreased and explained variance improved strongly |



### Longer Training Result: Selected Reward Combo

| Config | Steps | Envs | Steps/sec | Entropy | Approx KL | Value Loss | Explained Variance | Stability | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| hit_aggression_placement_combo | 10,000 | 4 | 35.8 | -3.292 | 0.00086 | 0.0486 | 0.00057 | Very stable | Preliminary updated-observation combo test |
| hit_aggression_placement_combo | 50,000 | 8 | 39.4 | -3.197 | 0.00423 | 0.0522 | 0.16569 | Stable | Longer training remains stable; explained variance improves, suggesting the critic starts learning the value function better |

The selected combined reward setting was trained for 50,000 steps using 8 parallel environments. The result remained stable, with `approx_kl=0.00423`, which is still below 0.01. Compared with the 10,000-step run, the value loss remained similar, while explained variance improved from approximately 0.00057 to 0.16569. This suggests that longer training helps the critic estimate the value function more effectively. Therefore, the selected reward setting is suitable for further longer training or final evaluation.

---

### Final Handoff-Aligned Reward Combo Training Result

| Config | Observation Version | Steps | Envs | Steps/sec | Entropy | Approx KL | Value Loss | Explained Variance | Stability | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| hit_aggression_placement_combo | Baseline observation | 50,000 | 8 | 39.4 | -3.197 | 0.00423 | 0.0522 | 0.16569 | Stable | Reward combo trained with baseline observation; useful as comparison but not final handoff-aligned setup |
| hit_aggression_placement_combo_heatmap | Heatmap + 16-frame stack | 50,000 | 8 | 37.1 | -3.216 | 0.00701 | 0.0389 | 0.76272 | Stable | Final reward combo using Siao Xuan’s updated observation setup; value loss decreased and explained variance improved strongly |
| hit_placement_heatmap | Heatmap + 16-frame stack | 50,000 | 8 | 36.0 | -3.255 | 0.0056 | 0.0471 | 0.4742 | Stable | Revised final reward-shaping candidate without aggression bonus. The model remained stable during 50k training, and the removal of aggression produced more controlled paddle movement during visual inspection. |
| hit_placement_heatmap | Heatmap + 16-frame stack | 100,000 | 8 | 40.5 | -3.098 | 0.0129 | 0.0532 | 0.4819 | Slightly unstable | Extended diagnostic training requested after the previous 0% evaluation. Explained variance remains similar to the 50k run, but approx KL increased above 0.01, suggesting PPO updates became less stable. This model should be evaluated visually and against square opponent before deciding whether it improves gameplay. |

The 100k extended training did not show clear improvement over the 50k model. Although explained variance increased slightly, the approximate KL rose above 0.01 and value loss also increased. This suggests that longer training did not clearly improve stability, and the 50k `hit_placement_heatmap` checkpoint remains the safer final reward-shaping candidate unless evaluation results show otherwise.

The final handoff-aligned reward configuration was trained using Siao Xuan’s heatmap observation setup with 16-frame stacking. Compared with the baseline-observation reward combo, the heatmap reward combo achieved a lower value loss and a much higher explained variance. Although the approximate KL increased to 0.00701, it remains below 0.01, indicating that PPO updates are still stable. Therefore, `hit_aggression_placement_combo_heatmap` is selected as the final reward-shaping candidate for evaluation and handoff.

### Debug Training Result: Hit + Placement Heatmap

| Config | Observation Version | Steps | Envs | Steps/sec | Entropy | Approx KL | Value Loss | Explained Variance | Stability | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| hit_placement_heatmap | Heatmap + 16-frame stack | 10,000 | 4 | 40.1 | -3.285 | 0.0050 | 0.0476 | 0.2049 | Stable | Revised reward variant without aggression bonus. Visual inspection showed more controlled paddle movement, better ball aiming, and fewer out-of-court hits compared with the previous aggression-based combo. |


---
## Evaluation 

##### N_MATCHES = 5
vs no_op  : 0 wins out of 5
vs random : 0 wins out of 5
vs square : 0 wins out of 5

### Evaluation Result: 50k Heatmap Reward Combo

| Config | Side | Opponent | Matches | Wins | Losses | Draws | Win Rate | Avg Point Diff | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| hit_aggression_placement_combo_heatmap | Right | no_op | 5 | 0 | 5 | 0 | 0.0% | N/A | Quick diagnostic evaluation |
| hit_aggression_placement_combo_heatmap | Right | random | 5 | 0 | 5 | 0 | 0.0% | N/A | Quick diagnostic evaluation |
| hit_aggression_placement_combo_heatmap | Right | square | 5 | 0 | 5 | 0 | 0.0% | N/A | Quick diagnostic evaluation |
| hit_aggression_placement_combo_heatmap | Right | square | 50 | 0 | 48 | 2 | 0.0% | -3.58 | Final evaluation; model did not meet the 85% square-opponent target |
| hit_placement_heatmap(50k) | Right | square | 5 | 1 | 3 | 1 | 20.0% | -2.2 | Quick evaluation. The no-aggression model improved compared with the previous aggression-based model, but visual inspection showed repeated paddle movement patterns and inconsistent ball tracking. |
| hit_placement_heatmap(100k) | Right | square | 5 | 5 | 0 | 0 | 100.0% | 18.6 | Quick evaluation for 100k checkpoint. The model achieved perfect win rate, but visual inspection showed a repeated opening-shot pattern and weaker response when the opponent started the rally. |
| hit_placement_heatmap(100k) | Right | square | 50 | 50 | 0 | 0 | 100.0% | 18.22 | Final right-side evaluation for 100k checkpoint. The model exceeded the 85% target, although visual inspection suggested a repeated opening-shot strategy rather than fully robust rally response. |
| hit_placement_heatmap(10k) | Left | square | 5 | 3 | 1 | 1 | 60.0% | 1.2 | Quick diagnostic after applying the left-side ball-position mirroring fix and retraining for 10k steps. The model showed improved left-side behaviour compared with the previous default-position rotation issue, but longer training is required for final evaluation. |
| hit_placement_heatmap (100k) | Left | square | 50 | 50 | 0 | 0 | 100.0% | 13.1 | Final left-side evaluation after retraining the 100k checkpoint with the fixed left-side ball-position mirroring wrapper. The model exceeded the 85% target against the square opponent. |
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
