# Phase 2: Parallel Experiment Execution

## Prerequisites

Phase 1 wiring must be committed and pushed before Members C and D can start.

---

## Member Assignments

### You (M1 Air, 2 envs) — Code Leader + Observation Variants

**Hardware:** M1 Air, `--envs 2`
**Total est. time:** ~25 min

```bash
python run_experiment.py --config baseline    --steps 10000 --envs 2
python run_experiment.py --config rgb_obs     --steps 10000 --envs 2
python run_experiment.py --config frame_diff  --steps 10000 --envs 2
python run_experiment.py --config heatmap_obs --steps 10000 --envs 2
python run_experiment.py --config stack_2     --steps 10000 --envs 2
```

> `rgb_obs`: FPS ~2× lower. `frame_diff`: near-zero frames. `stack_2`: slightly faster.

---

### Member B (M5 Pro, 8 envs) — PPO Hyperparameter Sweep

**Hardware:** M5 Pro, `--envs 8`
**Total est. time:** ~20 min

```bash
python run_experiment.py --config baseline     --steps 10000 --envs 8
python run_experiment.py --config entropy_0.0  --steps 10000 --envs 8
python run_experiment.py --config entropy_0.05 --steps 10000 --envs 8
python run_experiment.py --config large_batch  --steps 10000 --envs 8
python run_experiment.py --config small_batch  --steps 10000 --envs 8
python run_experiment.py --config lr_1e-4      --steps 10000 --envs 8
```

---

### Member C (Laptop, 2 envs) — Reward Shaping Variants

Must wait for Phase 1 wiring.

**Hardware:** Normal laptop, `--envs 2`
**Total est. time:** ~15 min

```bash
python run_experiment.py --config baseline         --steps 10000 --envs 2
python run_experiment.py --config aggression_bonus --steps 10000 --envs 2
python run_experiment.py --config hit_bonus_high   --steps 10000 --envs 2
```

---

### Member D (Laptop, 2 envs) — Training Strategy Variants

Must wait for Phase 1 wiring.

**Hardware:** Normal laptop, `--envs 2`
**Total est. time:** ~15 min

```bash
python run_experiment.py --config baseline      --steps 10000 --envs 2
python run_experiment.py --config no_curriculum --steps 10000 --envs 2
```

Additionally:
1. Run `train_selfplay_v2.py` to validate curriculum pipeline
2. Modify `train_selfplay_v2.py` to skip stages 1-2 and test no-curriculum selfplay

> **Note:** `no_curriculum` in `run_experiment.py` will match baseline since
> the harness has no curriculum (it lives in `train_selfplay_v2.py`).

---

### Members E+F (2 Laptops) — Phase 3 Prep Code (Parallel)

| Member | Task | File(s) | Est. |
|--------|------|---------|------|
| E | TwoTower CNN policy | `agent/two_tower_cnn.py` | 2-3 hrs |
| F | Grid search script | `scripts/grid_search.py` | 3-4 hrs |

---

## RESULTS.md Update Protocol

After each experiment, update the comparison table:

| Config | Steps | FPS | Entropy | Value Loss | Ep Len | Ep Rew | Machine |
|--------|-------|-----|---------|------------|--------|--------|---------|
| baseline | 6144 | 31.1 | -3.28 | 0.165 | 1.1k | -1.76 | M1 |

And paste full JSON in per-config section:

```json
{
  "experiment": "baseline",
  "description": "...",
  "timestamp": "...",
  "total_steps": 10000,
  "num_envs": 2,
  "elapsed_seconds": 320.5,
  "steps_per_second": 31.2
}
```

## Hardware Baseline Comparison

| Machine | Envs | Est. FPS | Est. Time for 10k |
|---------|------|----------|-------------------|
| M5 Pro | 8 | ~200+ | ~50s |
| M1 Air | 2 | ~31 | ~5 min |
| Laptop | 2 | ~20-30 | ~5-8 min |

## When Are Experiments "Done"?

1. Completes without errors
2. JSON result in `results/`
3. RESULTS.md table + full JSON section updated
4. FPS, entropy, value_loss, ep_rew_mean captured

No need for multiple runs — 10k steps is a smoke test for directional signals.
