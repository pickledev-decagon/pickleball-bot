# Experiment Results

> Cross-member comparison table. Each member adds their best row after each experiment.
> Detailed experiment logs in `progress/member-X-AREA.md`.

---

## Combined Results

| Member | Area | Branch | Config | Steps | Win Rate (right) | Win Rate (left) | Best Opponent | Avg Point Diff | Drive Link |
|--------|------|--------|--------|-------|-----------------|-----------------|--------------|---------------|------------|
| — | — | — | — | — | — | — | — | — | — |

*Rows sorted by average win rate (descending).*

---

## How to Add Your Result

After each experiment, add a row:

1. Run evaluation: use `eval_harness.py` or manual testing
2. Upload your best checkpoint to Google Drive
3. Add your row to the table above
4. Also update your detailed log in `progress/member-X-AREA.md`

### Win Rate Standards

| Opponent | Description |
|----------|-------------|
| `no_op` | Stationary opponent — always action [1,1,1] |
| `random` | Random actions — baseline sanity check |
| `square` | Square-motion pattern — curriculum Stage 2 opponent |
| `self (prev)` | Previous checkpoint — measures improvement over time |

---

## Phase 2 Legacy Results (May 13)

*Pre-overhaul experiments with 4-frame stack, no left/right training, no win-rate eval.*

| Experiment | Explained Variance | FPS | Notes |
|-----------|-------------------|-----|-------|
| baseline | 0.009 | 32.1 | Default grayscale |
| heatmap_obs | **0.329** | 33.3 | **Best Phase 2 result** |
| rgb_obs | 0.004 | 24.9 | 3x channels, slower |
| frame_diff | -0.002 | 18.5 | Motion-based, slower |
| stack_2 | 0.006 | 33.0 | Less temporal info |

*Full Phase 2 details: `BUILD_LOG.md` → Phase 2 section*
