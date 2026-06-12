# dPickleBall Bot 🤖🏓

**Pickleball RL agent** trained via self-play and visual observation for the dPickleBall competition. Uses SB3 PPO with 16-frame stacked grayscale observations, ball-tracking heatmaps, and a 3-stage curriculum (no-op → square motion → self-play).

---

## 👥 Team

| Member | Branch | Area | Files |
|--------|--------|------|-------|
| **Amber** | `team/arch-variants` | Model Architecture | `agent/two_tower_cnn.py`, `env/wrapper.py` (preprocessing pipeline) |
| **Yenly** | `team/reward-shaping` | Reward Functions | `agent/reward_shaping.py`, `agent/experiment_configs.py` |
| **Siao Xuan** | `team/obs-variants` | Observation | `env/preprocessing.py`, ball detection + heatmap (via `env/wrapper.py`) |
| **Keven** | `team/hyperparams` | Hyperparameters | `sweep_configs/*.json`, `scripts/grid_search.py` |
| **Hui Xuan** | `team/training-strategy` | Training Strategy | `agent/train_selfplay_v2.py`, `agent/selfplay_callback.py`, `agent/buffer.py`, `config.py` |

> **`env/wrapper.py` note:** Amber owns the preprocessing pipeline (frame conversion, resize, grayscale). Siao Xuan owns ball detection + heatmap rendering consumed by the wrapper. Coordinate changes via PR.

### Workflow

```
Everyone experiments on their branch → push code, checkpoints to Google Drive
                ↓
Discuss results → pick best config for each area
                ↓
Keven does final combined training on main
                ↓
Package for competition → tag and submit
```

---

## 📖 What to Read First (in Order)

| # | File | Why |
|---|------|-----|
| **1** | [`handoff.md`](handoff.md) | Copy-paste to ChatGPT/Copilot so your LLM understands the full project context. |
| **2** | [`CONTRIBUTING.md`](CONTRIBUTING.md) | One-stop shop: roles, setup, git workflow, training plan, quick reference. |
| **3** | `progress/member-X-AREA.md` | YOUR assigned area — variants, done criteria, blockers, dependencies, next steps. |
| **4** | [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md) | Technical deep dive — read sections relevant to your area as needed. |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10, conda (Miniconda recommended)
- **Mac:** Unity build `.app` (see step 3)
- **Windows:** Unity build `.exe` (see [Windows setup notes](#windows-setup-notes))

### 1. Clone the repo

```bash
git clone https://github.com/pickledev-decagon/pickleball-bot.git
cd pickleball-bot
git submodule update --init --depth 1
```

### 2. Create environment and install

```bash
conda create -n dpickleball pip python=3.10.12 -y
conda activate dpickleball

# ML-Agents (from submodule)
pip install -e ./dpickleball-ml-agents/ml-agents-envs

# Core packages
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

### 3. Download Unity build and set path

Download from [Google Drive](https://drive.google.com/drive/folders/1lFqj6lopoIO96C_IO8yGXGzRMY1S4fzi) (395 MB).

Create a `.env` file in the project root:

**Mac:**
```bash
echo 'DPICKLEBALL_BUILD_PATH="/path/to/Training/Mac/Mac.app"' > .env
chmod -R 755 /path/to/Training/Mac/Mac.app
```

**Windows:**
```bash
echo DPICKLEBALL_BUILD_PATH=C:\path\to\Training\Windows\dp.exe > .env
```

### 4. Verify

```bash
python run_experiment.py --list
python run_experiment.py --config baseline --steps 2000 --envs 2
```

### 5. Create your branch and start

```bash
git checkout -b team/YOUR-AREA   # e.g. team/reward-shaping
# ... experiment, log, commit, push ...
```

---

## <a name="windows-setup-notes"></a>🪟 Windows Setup Notes

| Area | Mac | Windows |
|------|-----|---------|
| Unity build | `Training/Mac/Mac.app` | `Training/Windows/dp.exe` |
| `BUILD_PATH` in `.env` | `/path/to/Mac.app` | `C:\path\to\dp.exe` |
| Kill Unity processes | `pkill -f "Mac.app"` | `taskkill /F /IM dp.exe` |
| Free ports | `lsof -ti:5005-5012 \| xargs kill` | `netstat -ano \| findstr :5005` (PID is last column) → `taskkill /F /PID <id>` |
| Conda | Same commands (cross-platform) | Same commands |
| Python packages | Same `pip install` | Same `pip install` |

All `pip install` commands and Python code work identically on Windows. The only differences are the Unity build path format and process management commands.

---

## 📋 Full Guides

| Guide | What it covers |
|-------|---------------|
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Team roles, initial setup, daily git workflow, training plan, experiment protocol, PR process, troubleshooting |
| [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md) | Architecture deep dive, reward formulas, curriculum stages, self-play mechanics, PPO config, reading training output, experiment catalog with all variants and decision tree |
| [`handoff.md`](handoff.md) | LLM-ready project context — copy-paste to ChatGPT/Copilot for instant understanding |

---

## 📊 Track Progress

| File | What it shows |
|------|---------------|
| [`DASHBOARD.md`](DASHBOARD.md) | Full milestone tracker — what's done, what's next |
| [`PROGRESS.md`](PROGRESS.md) | Team progress bars — per-member completion at a glance |
| [`RESULTS.md`](RESULTS.md) | Cross-member comparison table — best configs per area |
| [`progress/README.md`](progress/README.md) | How to log experiments and use the tracking system |
| `progress/member-1-architecture.md` | M1 experiment log |
| `progress/member-2-reward.md` | M2 experiment log |
| `progress/member-3-observation.md` | M3 experiment log |
| `progress/member-4-hyperparameters.md` | M4 experiment log |
| `progress/member-5-training-strategy.md` | M5 experiment log |

---

## 🏗️ Repo Structure

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
├── sweep_configs/          # Grid search configs (frame_stack=16)
├── progress/               # Per-member experiment logs
├── archive/                # Historical docs (build log, phase plans)
├── config.py               # Central config (paths, defaults)
├── run_experiment.py       # Single experiment runner
├── dpickleball-ml-agents/  # Git submodule (forked ML-Agents)
├── handoff.md              # LLM context for teammates
├── CONTRIBUTING.md         # ← start here
├── PROJECT_GUIDE.md        # Technical reference
├── RESULTS.md              # Experiment comparison table
├── DASHBOARD.md            # Milestone tracker
└── PROGRESS.md             # Progress bars
```

---

## 📈 Project Status

| Phase | Status |
|-------|--------|
| Environment setup | ✅ Done |
| Pipeline + all fixes | ✅ Done |
| B1 smoke test (10k steps) | ✅ Passed |
| **Team experiments** | ⏳ Ready to start |
| Combination testing | ⏳ Not started |
| Final combined training | ⏳ Not started |
| Competition submission | ⬜ Not started |

See **[DASHBOARD.md](DASHBOARD.md)** for the full milestone tracker.
