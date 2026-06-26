pickleball_bot — CLAUDE.md

## Training Guardrail
**DO NOT start any training without explicit user approval.**
This includes running `train_selfplay_v2.py`, `run_experiment.py`, `scripts/grid_search.py`,
or any script that calls `model.learn()`. Before any training run, the user must explicitly
confirm all hyperparameters, environment configuration, and the intended experiment goal.

## Key Project Context
- **RL algorithm**: PPO (Stable-Baselines3) with CnnPolicy
- **Environment**: Unity ML-Agents pickleball simulator
- **Action space**: MultiDiscrete([3,3,3]) = [vertical, horizontal, rotation]
- **Observation**: 16-frame stack of 84x168 grayscale with ball-tracking heatmap
- **Training**: 200k total steps across 8 parallel envs (4 left + 4 right)
- **Curriculum**: Stage 1 (0-30%) no-op opponent, Stage 2 (30-100%) square-motion opponent
- **Team**: 5 members — Keven (hyperparams), Yenly (rewards), Siao Xuan (obs),
  Amber (model), Hui Xuan (training strategy)

## Heatmap Sync Warning
The `_render_heatmap()` method exists in BOTH `env/wrapper.py` AND
`agent/competition_export.py`. These MUST stay identical. If you change one,
change the other. A mismatch will silently break model performance.
