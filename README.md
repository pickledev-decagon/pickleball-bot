# dPickleBall Bot

Pickleball RL agent trained via self-play and visual observation for a course competition. Uses SB3 PPO, 4-stack grayscale frames, and a 3-stage curriculum (no-op → square motion → self-play).

## Quick Start

```bash
conda activate dpickleball
python run_experiment.py --config baseline --steps 10000 --envs 2
```

See [SETUP_TEAMMATE.md](SETUP_TEAMMATE.md) for full environment setup.

## Documentation

| File | What it covers |
|---|---|
| [PROJECT_GUIDE.md](PROJECT_GUIDE.md) | Architecture, reward formulas, training curriculum, metrics interpretation |
| [OVERVIEW.md](OVERVIEW.md) | One-page reference of decisions and structure |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Experiment catalog with hypotheses, decision tree, and results |
| [SETUP_TEAMMATE.md](SETUP_TEAMMATE.md) | Step-by-step setup for teammate laptops |
| [RESULTS.md](RESULTS.md) | Running comparison table of all experiments |
| [BUILD_LOG.md](BUILD_LOG.md) | Chronological build history and resolved issues |

## Key Files

| Path | Role |
|---|---|
| `env/preprocessing.py` | BallTracker (HSV detection) |
| `env/wrapper.py` | `EnhancedUnityWrapper` (gymnasium Env) |
| `agent/wrapper_vec.py` | 8× parallel env factory |
| `agent/buffer.py` | OpponentBuffer (checkpoint pool) |
| `agent/reward_shaping.py` | Dense reward + paddle-hit detection |
| `agent/train_selfplay_v2.py` | 3-stage curriculum training loop |
| `agent/evaluate.py` | Win-rate measurement |
| `run_experiment.py` | Named experiment harness |
| `agent/experiment_configs.py` | 13 experiment presets |
