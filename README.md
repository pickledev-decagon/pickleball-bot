# dPickleBall Bot 🤖🏓

**Pickleball RL agent** trained via self-play and visual observation for the dPickleBall competition. Uses SB3 PPO with 4-frame stacked grayscale observations, ball-tracking heatmaps, and reward shaping with annealed dense rewards. **Final result: 100% win rate** across all opponent types on both court sides (see [REPORT.md](REPORT.md)).

---

## 🏆 Final Results

The submitted checkpoint is `Competition_Group(10)/exp003_75000_steps.zip` (26.2 MB, SB3 PPO, deterministic inference).

**Evaluation — 50 matches per condition:**

| Condition | Win Rate | Avg Point Diff |
|-----------|----------|----------------|
| Right side vs. square | 100% | +15.88 |
| Left side vs. square | 100% | +17.78 |
| Right / Left vs. no_op | 100% | — |

**Inference latency:** mean 1.48 ms / p95 2.21 ms per step — far within the 0.01 s competition deadline.

**Submission entry points** (self-contained, no project imports; model loaded from same directory):

| File | Role |
|------|------|
| `teamX.py` | Left-side agent — mirrors observations & actions |
| `teamY.py` | Right-side agent |
| `exp003_75000_steps.zip` | PPO checkpoint (`MultiInputActorCriticPolicy`) |
| `Competition_local_mac.py` | Local match runner for pre-submission testing |

## 👥 Team

| Member | Area | Key Contribution |
|--------|------|-----------------|
| **Amber** | Model Architecture | TwoTowerCNN / SpatialSoftmaxCNN alternatives to NatureCNN |
| **Yenly** | Reward Shaping | 18 sensitivity experiments → hit 1.0 + placement 0.025 optimal |
| **Siao Xuan** | Observation | HSV ball tracking, Gaussian heatmap, frame stacking, vector obs |
| **Keven** | Hyperparameters & Infra | PPO config, experiment harness (`run_experiment.py`), evaluation |
| **Hui Xuan** | Training Strategy | Curriculum schedule, self-play buffer, `train_selfplay_v2.py` |

## 🔬 Method Summary

- **Observation:** dict input — `visual (4, 84, 168)` grayscale + Gaussian heatmap overlay (σ=5, w=0.7) at the tracked ball position, plus a 6-D `vector` observation `[ball_vx, ball_vy, paddle_x, paddle_y, ball_x, ball_y]`. Heatmap was the single biggest improvement (explained variance 0.33 vs. 0.009 without).
- **Rewards:** sparse win/loss from Unity + dense shaping (proximity, hit bonus, placement bonus) annealed to zero over the first 60% of training.
- **Training:** PPO, 8 parallel Unity envs (4 left + 4 right), 200k-step 2-stage curriculum (no_op → square opponent).
- **Key bugs fixed:** double normalization, 1D hit detection, disabled rotation penalty, spinning square opponent.

Full details: **[REPORT.md](REPORT.md)** · training data flow: **[FLOW.md](FLOW.md)**

## 📖 Reading Order

| # | File | What it covers |
|---|------|---------------|
| 1 | **[`FLOW.md`](FLOW.md)** | Full system data-flow diagrams (Unity → wrapper → PPO → export) |
| 2 | **[`REPORT.md`](REPORT.md)** | Final report — methodology, results, lessons learned |
| 3 | **[`PLAN.md`](PLAN.md)** | Experiment planning & validation methodology |
| 4 | `archive/` | Historical build log & phase docs |

## 🚀 Quick Start (Reproduction)

### Prerequisites

- Python 3.10, conda (Miniconda recommended)
- Unity build `.app`/`.exe` ([Google Drive download](https://drive.google.com/drive/folders/1lFqj6lopoIO96C_IO8yGXGzRMY1S4fzi), 395 MB)

### 1. Clone

```bash
git clone https://github.com/pickledev-decagon/pickleball-bot.git
cd pickleball-bot
git submodule update --init --depth 1
```

### 2. Environment setup

```bash
conda create -n dpickleball pip python=3.10.12 -y
conda activate dpickleball

pip install -e ./dpickleball-ml-agents/ml-agents-envs
pip install stable-baselines3 opencv-python==4.7.0.72 \
            matplotlib numpy "torch>=2.3,<2.6" "torchvision>=0.18,<0.20"
```

<details>
<summary>💡 If you get numpy version errors</summary>

```bash
pip install "numpy<2,>=1.23.5"
pip install "setuptools<70"
```
</details>

### 3. Unity build path

Create a `.env` in the project root:

**Mac:**
```bash
echo 'DPICKLEBALL_BUILD_PATH="/path/to/Mac/Mac.app"' > .env
```

**Windows:**
```bash
echo DPICKLEBALL_BUILD_PATH=C:\path\to\dp.exe > .env
```

### 4. Verify

```bash
python run_experiment.py --list
python run_experiment.py --config baseline --steps 2000 --envs 2
```

## 🛠️ Usage

| Task | Command |
|------|---------|
| Train (experiment harness) | `python run_experiment.py --config <name> --steps 100000 --envs 8` |
| Curriculum / self-play training | `python agent/train_selfplay_v2.py` |
| Evaluate a checkpoint | `python evaluate_checkpoint.py <checkpoint.zip> --matches 20` |
| Quick eval vs all opponents | `python eval_quick.py` |
| Watch agent play live | `python showcase.py --opponent square` |
| Local competition match | `python Competition_local_mac.py` |

## 🏗️ Repo Structure

```
pickleball-bot/
├── Competition_Group(10)/  # Competition submission (teamX.py, teamY.py, checkpoint zip)
├── Competition_local_mac.py # Local competition match runner
├── agent/                  # Training logic
│   ├── buffer.py           # Opponent buffer (checkpoint pool)
│   ├── competition_export.py # Async inference for competition
│   ├── eval_harness.py     # Win-rate evaluation
│   ├── experiment_configs.py # 43 experiment presets
│   ├── reward_shaping.py   # Proximity, hit, placement rewards
│   ├── selfplay_callback.py # Mid-training checkpoint → buffer
│   ├── train_selfplay_v2.py # Curriculum training
│   ├── two_tower_cnn.py    # TwinNatureCNN, SpatialSoftmaxCNN
│   └── wrapper_vec.py      # 8 parallel envs (4 left + 4 right)
├── env/                    # Unity environment side
│   ├── preprocessing.py    # BallTracker, SmoothBallTracker
│   ├── wrapper.py          # EnhancedUnityWrapper
│   └── *_opponent.py       # Specialist / self-play / ball-tracking opponents
├── scripts/                # Benchmarking, validation, graph generation
├── sweep_configs/          # Grid search phase configs
├── archive/                # Historical docs (build log, phase plans)
├── results/                # Final evaluation graphs & validation reports
├── config.py               # Central config (paths, defaults)
├── run_experiment.py       # Single experiment runner
├── evaluate_checkpoint.py  # Checkpoint evaluation with match metrics
├── showcase.py             # Watch agent play live
├── FLOW.md / PLAN.md / REPORT.md  # Docs
└── dpickleball-ml-agents/  # Git submodule (forked ML-Agents)
```

> **Note:** `checkpoints_selfplay_v2/`, `logs_selfplay_v2/`, large files in `results/`, and the Unity build are gitignored — download the Unity build separately (link above). Checkpoints live on Google Drive / local only.

---

## 🪟 Windows Setup Notes

| Area | Mac | Windows |
|------|-----|---------|
| Unity build | `Training/Mac/Mac.app` | `Training/Windows/dp.exe` |
| `BUILD_PATH` in `.env` | `/path/to/Mac.app` | `C:\path\to\dp.exe` |
| Kill Unity processes | `pkill -f "Mac.app"` | `taskkill /F /IM dp.exe` |
| Free ports | `lsof -ti:5005-5012 \| xargs kill` | `netstat -ano \| findstr :5005` (PID is last column) → `taskkill /F /PID <id>` |
| Conda | Same commands (cross-platform) | Same commands |
| Python packages | Same `pip install` | Same `pip install` |

All `pip install` commands and Python code work identically on Windows. The only differences are the Unity build path format and process management commands.
