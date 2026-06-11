# dPickleBall Bot — Quick Reference

**Goal:** Best pickleball RL agent for course competition.

## Strategy Decisions

| Decision | Value | Rationale |
|---|---|---|
| Algorithm | PPO (SB3 2.8.0) | Proven, stable, includes 37 impl details (ICLR 2022) |
| Observation | 4 stacked grayscale frames `(4,84,168)` | Motion perception, SB3-compatible |
| Ball tracking | HSV thresholding `H=[20,35] S=[80,255] V=[80,255]` | 98-100% detection rate on live frames |
| Parallelism | 8× DummyVecEnv via Unity (in-process) | Champion feedback: 8 is sweet spot |
| Curriculum | 3 stages: no-op → square → self-play | Lecturer feedback: curriculum learning helps |
| Rewards | Dense (proximity + hit) → annealed to zero by 60% | Sparse reward problem |
| Device | CPU | Unity rendering (20ms) dwarfs inference (1-2ms) |
| Self-play | OpponentBuffer (max 5, random sample) | OpenAI 2017: prevents overfitting |

## Project Structure

```
pickleball_bot/
├── env/
│   ├── preprocessing.py       BallTracker (HSV), frame_diff, detect_paddles
│   └── wrapper.py             EnhancedUnityWrapper (gymnasium Env, 4-stack)
├── agent/
│   ├── wrapper_vec.py         8× parallel Unity env factory
│   ├── buffer.py              OpponentBuffer (checkpoint pool)
│   ├── reward_shaping.py      Dense reward, hit detection, annealing
│   ├── train_selfplay_v2.py   3-stage curriculum training
│   ├── evaluate.py            Win-rate evaluation
│   └── experiment_configs.py  13 experiment presets
├── results/                   Experiment output JSONs
├── archive/                   v1 (pre-curriculum, single-env) files
├── README.md                  Entry point
├── PROJECT_GUIDE.md           Full technical guide
├── EXPERIMENTS.md             Experiment catalog + decision tree
├── SETUP_TEAMMATE.md          Teammate laptop setup
├── RESULTS.md                 Experiment comparison table
└── BUILD_LOG.md               Chronological build history
```

## Competition Constraints

| Rule | Implication |
|---|---|
| 0.01s max response time | Use async inference (model runs while prev action executes) |
| No extra Python packages | Only numpy, opencv, torch — already installed |
| Left + right paddle both AI | Must train both sides |
| Visual obs only | `84×168` RGB camera, no state vectors |
| First to 21 points | ~500-2000 steps per episode |

## Quick Commands

```bash
conda activate dpickleball
python run_experiment.py --list                                          # See available configs
python run_experiment.py --config baseline --steps 10000 --envs 2        # Run baseline (M1)
python run_experiment.py --config entropy_0.05 --steps 20000 --envs 8    # Full run (M5 Pro)
tensorboard --logdir results/                                            # Monitor training
python agent/evaluate.py --checkpoint opponent_buffer/curriculum_final_*.zip  # Evals
```
