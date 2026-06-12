# 📁 Per-Member Progress Tracking

This directory tracks each team member's experiments, decisions, blockers, and next steps.

## How to Use

### Every Member Does This

1. **Start an experiment** → Add row to your `member-X-AREA.md` experiment log with status "Running"
2. **Finish an experiment** → Fill in win rates, upload checkpoint to Drive, add link
3. **Update RESULTS.md** in the project root with your best result row
4. **Hit a blocker** → Add to your blocker log in your progress file
5. **Keven reviews PROGRESS.md** in the project root daily

### File Structure

| File | Who Owns | What It Tracks |
|------|----------|---------------|
| `member-1-architecture.md` | Amber | CNN architectures, feature extractors |
| `member-2-reward.md` | Yenly | Reward shaping variants, weights |
| `member-3-observation.md` | Siao Xuan | Frame stack, heatmap, frame diff, preprocessing |
| `member-4-hyperparameters.md` | Keven | Learning rate, entropy, batch size sweeps |
| `member-5-training-strategy.md` | Hui Xuan | Curriculum stages, buffer strategy, checkpointing |

### Typical Experiment Cycle

```
Pick config → run_experiment.py → log results → update progress file → update RESULTS.md
```

### What Goes in RESULTS.md vs Progress Files

| RESULTS.md (project root) | member-X-AREA.md (this folder) |
|--------------------------|-------------------------------|
| Best row per member | All experiments (successes + failures) |
| Comparison across areas | Detailed notes, decisions, blockers |
| Used for final config selection | Personal journal for the member |
