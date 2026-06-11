# dPickleBall Bot 🤖🏓

**Pickleball RL agent** trained via self-play and visual observation for the dPickleBall competition. Uses SB3 PPO with 16-frame stacked grayscale observations, ball-tracking heatmaps, and a 3-stage curriculum (no-op → square motion → self-play).

---

## 👥 Team Workflow

**5 members, 5 areas, 1 shared repo.** Everyone works in parallel on their own branch.

| Member | Branch | Area | Files |
|--------|--------|------|-------|
| **Keven** (you) | `team/arch-variants` | Model Architecture | `agent/two_tower_cnn.py`, `env/wrapper.py` |
| **Member 2** | `team/reward-shaping` | Reward Functions | `agent/reward_shaping.py`, `agent/experiment_configs.py` |
| **Member 3** | `team/obs-variants` | Observation | `env/preprocessing.py` |
| **Member 4** | `team/hyperparams` | Hyperparameters | `sweep_configs/*.json`, `scripts/grid_search.py` |
| **Member 5** | `team/training-strategy` | Training Strategy | `agent/train_selfplay_v2.py`, `agent/selfplay_callback.py`, `agent/buffer.py`, `config.py` |

### Flow

```
Everyone experiments on their branch → push code to GitHub, checkpoints to Google Drive
                ↓
Discuss results as a team → pick best config for each area
                ↓
One person (Keven) does final combined training on main
                ↓
Package for competition submission → tag and submit
```

### Repo URL

```
https://github.com/pickledev-decagon/pickleball-bot
```

---

## 🚀 Quick Start for Teammates

### 1. Clone the repo

```bash
git clone https://github.com/pickledev-decagon/pickleball-bot.git
cd pickleball-bot
git submodule update --init --depth 1
```

### 2. Install dependencies

```bash
# ML-Agents (from submodule)
pip install -e ./dpickleball-ml-agents/ml-agents-envs

# Core packages
conda create -n dpickleball pip python=3.10.12
conda activate dpickleball
pip install stable-baselines3 opencv-python==4.7.0.72 numpy==1.26.4 torch
```

### 3. Set your Unity build path

Create a `.env` file in the project root:

```bash
echo 'DPICKLEBALL_BUILD_PATH="/absolute/path/to/your/Unity/Build.app"' > .env
```

### 4. Create your branch and start working

```bash
git checkout -b team/YOUR-AREA   # e.g. team/reward-shaping
# ... make changes, train, experiment ...
git add <your-files>
git commit -m "Describe your change"
git push -u origin team/YOUR-AREA
```

---

## 📋 Full Team Workflow Guide

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for:
- Daily git workflow
- How to open a Pull Request
- What to commit vs what goes to Google Drive
- How to record results
- Merge process

---

## 📁 Repo Structure

```
pickleball-bot/
├── agent/                  # Training logic
│   ├── buffer.py           # Opponent buffer (checkpoint pool)
│   ├── competition_export.py # Async inference for competition
│   ├── eval_harness.py     # Win-rate evaluation
│   ├── evaluate.py         # Legacy (re-exports from competition_export)
│   ├── experiment_configs.py # 15+ experiment presets
│   ├── reward_shaping.py   # Proximity, hit, placement, aggression rewards
│   ├── selfplay_callback.py # Mid-training checkpoint → buffer
│   ├── train_selfplay_v2.py # 3-stage curriculum training
│   ├── two_tower_cnn.py    # TwinNatureCNN, SpatialSoftmaxCNN
│   └── wrapper_vec.py      # 8 parallel envs (4 left + 4 right)
├── env/                    # Unity environment wrapper
│   ├── preprocessing.py    # BallTracker, SmoothBallTracker
│   └── wrapper.py          # EnhancedUnityWrapper (left/right support)
├── scripts/                # Utilities
│   ├── grid_search.py      # Hyperparameter sweep
│   ├── test_competition_export.py # Export dry-run
│   └── validate_tracker.py # Ball tracker validation
├── sweep_configs/          # Grid search configs
├── config.py               # Central config (paths, defaults)
├── run_experiment.py       # Single experiment runner
├── dpickleball-ml-agents/  # Git submodule (forked ML-Agents)
├── RESULTS.md              # Experiment results table
├── BUILD_LOG.md            # Chronological build history
└── CONTRIBUTING.md         # Team workflow guide
```

---

## 📊 Project Status

| Phase | Status |
|-------|--------|
| Environment setup | ✅ Done (May 4) |
| Pipeline + all fixes | ✅ Done |
| Git repo initialized | ✅ Done |
| Validation tests (A1, A2) | ✅ Passed |
| **Training runs** | ⏳ Waiting for team |
| Competition submission | ⬜ Not started |

See **[DASHBOARD.md](DASHBOARD.md)** for full milestone tracker.

---

## 📚 Documentation

| File | What it covers |
|------|---------------|
| [CONTRIBUTING.md](CONTRIBUTING.md) | Team git workflow, setup, etiquette |
| [PROJECT_GUIDE.md](PROJECT_GUIDE.md) | Architecture, reward formulas, training curriculum, metrics |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Experiment catalog with hypotheses and decision tree |
| [RESULTS.md](RESULTS.md) | Running comparison table of all experiments |
| [BUILD_LOG.md](BUILD_LOG.md) | Chronological build history and resolved issues |
| [DASHBOARD.md](DASHBOARD.md) | Milestone tracker and progress bar |
| [SETUP_TEAMMATE.md](SETUP_TEAMMATE.md) | Legacy setup guide |
