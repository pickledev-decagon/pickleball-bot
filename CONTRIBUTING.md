# Contributing Guide — dPickleBall Bot

## 📋 Team Roles

| Member | Branch | Area | Files They Own |
|--------|--------|------|---------------|
| **Member 1** | `team/arch-variants` | Model Architecture | `agent/two_tower_cnn.py`, `env/wrapper.py` (preprocessing pipeline: frame conversion, resize, grayscale) |
| **Member 2** | `team/reward-shaping` | Reward Functions | `agent/reward_shaping.py`, `agent/experiment_configs.py` |
| **Member 3** | `team/obs-variants` | Observation | `env/preprocessing.py` (BallTracker, heatmap, frame diff), ball detection + heatmap rendering (via `env/wrapper.py`) |
| **Member 4** | `team/hyperparams` | Hyperparameters | `sweep_configs/*.json`, `scripts/grid_search.py` |
| **Member 5** | `team/training-strategy` | Training Strategy | `agent/train_selfplay_v2.py`, `agent/selfplay_callback.py`, `agent/buffer.py`, `config.py` |

**Note on `env/wrapper.py`:** M1 owns the observation preprocessing pipeline (frame conversion, resize, grayscale) inside this file. M3 owns the ball detection and heatmap rendering logic that the wrapper consumes. Both coordinate changes to shared interfaces via PR.

**Rule:** Only modify files in your assigned area. If you need a change in a shared file, open a small PR targeting `main` first.

---

## 📖 Quick Reference

### Strategy Decisions

| Decision | Value | Rationale |
|----------|-------|-----------|
| Algorithm | PPO (SB3 2.8.0) | Proven, stable, includes 37 implementation details |
| Observation | 16 stacked grayscale frames `(16,84,168)` | Motion perception + temporal context |
| Ball tracking | HSV thresholding `H=[20,35] S=[80,255] V=[80,255]` | 98–100% detection rate |
| Parallelism | 8× DummyVecEnv via Unity (in-process) | 8 is sweet spot for time/throughput |
| Curriculum | 3 stages: no-op → square → self-play | Curriculum learning helps sparse-reward tasks |
| Rewards | Dense (proximity + hit) → annealed to zero by 60% | Avoids sparse-reward problem |
| Device | CPU | Unity rendering (20ms) dwarfs inference (1–2ms) |
| Self-play | OpponentBuffer (max 5, random sample) | Prevents overfitting to a single style |

### Project Structure

```
pickleball_bot/
├── env/
│   ├── preprocessing.py        BallTracker (HSV), frame_diff, detect_paddles
│   └── wrapper.py              EnhancedUnityWrapper (gymnasium Env, 16-stack)
├── agent/
│   ├── wrapper_vec.py          8× parallel Unity env factory
│   ├── buffer.py               OpponentBuffer (checkpoint pool)
│   ├── reward_shaping.py       Dense reward, hit detection, annealing
│   ├── train_selfplay_v2.py    3-stage curriculum training
│   ├── evaluate.py             Win-rate evaluation
│   └── experiment_configs.py   Named experiment presets
├── sweep_configs/              Grid search sweep configs
├── scripts/grid_search.py      Hyperparameter sweep runner
├── results/                    Experiment output JSONs
├── progress/                   Per-member experiment logs
├── archive/                    Historical planning docs
├── CONTRIBUTING.md             ← this file
├── PROJECT_GUIDE.md            Full technical deep dive
├── FLOW.md                     Architecture diagram + data shapes
├── DASHBOARD.md                Milestone tracker
├── PROGRESS.md                 Team progress bars
├── RESULTS.md                  Cross-member comparison table
└── handoff.md                  LLM context for team members
```

### Competition Constraints

| Rule | Implication |
|------|-------------|
| 0.01s max response time | Use async inference (model runs while prev action executes) |
| No extra Python packages | Only numpy, opencv, torch — already installed |
| Left + right paddle both AI | Must train and evaluate both sides |
| Visual obs only | `84×168` RGB camera, no state vectors |
| First to 21 points | ~500–2000 steps per episode |

---

## 🖥️ Initial Setup (Once)

Estimated time: 10–15 minutes. Do this once per machine.

### Step 1: Create Conda Environment

```bash
conda create -n dpickleball pip python=3.10.12 -y
conda activate dpickleball
```

### Step 2: Install ML-Agents Fork

```bash
git clone https://github.com/dPickleball/dpickleball-ml-agents.git
pip install -e ./dpickleball-ml-agents/ml-agents-envs
pip install -e ./dpickleball-ml-agents/ml-agents
```

(If you cloned the project repo first, the submodule is already at `dpickleball-ml-agents/` — just run the `pip install` lines above from the project root.)

### Step 3: Install Dependencies

```bash
pip install stable-baselines3 opencv-python==4.7.0.72 \
            matplotlib numpy "torch>=2.3,<2.6" "torchvision>=0.18,<0.20"
```

If you get numpy version errors:
```bash
pip install "numpy<2,>=1.23.5"
pip install "setuptools<70"
```

### Step 4: Download Unity Build

1. Go to: https://drive.google.com/drive/folders/1lFqj6lopoIO96C_IO8yGXGzRMY1S4fzi
2. Download `dPickleball BuildFiless.zip` (395 MB)
3. Unzip to e.g. `~/Desktop/dp_build/`
4. Check the path contains `Training/` with your platform folder

**Mac:**
```bash
chmod -R 755 ~/Desktop/dp_build/Training/Mac/Mac.app
echo 'DPICKLEBALL_BUILD_PATH="/path/to/Training/Mac/Mac.app"' > .env
```

**Windows:**
```bash
echo DPICKLEBALL_BUILD_PATH=C:\path\to\Training\Windows\dp.exe > .env
```

### Step 5: Verify Everything Works

```bash
python run_experiment.py --list

# Quick test (2 minutes, 2 parallel envs):
python run_experiment.py --config baseline --steps 2000 --envs 2
```

### `run_experiment.py` CLI Reference

```bash
python run_experiment.py <flags>
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config <name>` | `baseline` | Experiment preset. See `--list` for options. |
| `--steps <N>` | `10000` | Total timesteps across all envs |
| `--envs <N>` | `2` | Number of parallel Unity environments |
| `--list` | — | Print available configs and exit |

### Output Format

Each run produces a JSON file in `results/`:

```json
results/baseline_20260513_163541.json
{
  "experiment": "baseline",
  "description": "Current default: grayscale, 4-stack, entropy 0.01...",
  "timestamp": "20260513_163541",
  "total_steps": 6144,
  "num_envs": 2,
  "elapsed_seconds": 197.9,
  "steps_per_second": 31.1,
  "train/entropy": -3.28,
  "train/policy_loss": 0.0081,
  "train/value_loss": 0.165,
  "train/approx_kl": 6.11e-6
}
```

After each run, paste the JSON into `RESULTS.md` under the corresponding section.

### Cleanup (If Experiments Crash)

**Mac:**
```bash
# Kill all Unity processes
pkill -f "Mac.app/Contents/MacOS/dp"

# Free up ports (5005–5012 range)
lsof -ti:5005-5012 | xargs kill -9 2>/dev/null || true
```

**Windows:**
```cmd
:: Kill all Unity processes
taskkill /F /IM dp.exe

:: Free up ports (5005–5012 range)
netstat -ano | findstr :5005
:: Then: taskkill /F /PID <id>
```

Add an alias to your shell config:

**Mac (`~/.zshrc`):**
```bash
alias killdp='pkill -f "Mac.app/Contents/MacOS/dp"'
```

**Windows (PowerShell profile):**
```powershell
function killdp { taskkill /F /IM dp.exe }
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| `Permission denied` on Unity build | `chmod -R 755 /path/to/Mac.app` |
| `_ARRAY_API not found` | `pip install "numpy<2"` |
| `gym` / `gymnasium` conflicts | `pip install gymnasium && pip uninstall gym -y` |
| `No module named 'mlagents_envs'` | Did you install the ML-Agents fork? See Step 2. |
| `Address already in use` | Unity port still occupied. Run cleanup commands above. |
| Python version wrong | `conda activate dpickleball && python --version` should show `3.10.x` |
| `tensorboard` not found | `pip install tensorboard` |
| Unity window flashing on/off | Normal — gRPC connection establishing. Wait 3–5 seconds. |
| Very slow step time (>50ms) | Close other GPU-intensive apps. Reduce `--envs`. |

---

## 🔄 Daily Workflow

```bash
# 1. Get the latest main
git checkout main
git pull
git submodule update --init --depth 1   # update ml-agents if changed

# 2. Switch to your branch and merge latest main
git checkout team/YOUR_BRANCH
git merge main

# 3. Work, commit, push
git add <your-files>
git commit -m "Describe what you changed and why"
git push -u origin team/YOUR_BRANCH

# 4. (Optional) Open a Pull Request on GitHub when your piece is ready
#    Keven reviews and clicks "Squash and merge"
```

---

## 📤 What to Push to GitHub vs Google Drive

### Push to GitHub (small files)
- Code changes (`.py` files) — these are the actual work
- Experiment results in `RESULTS.md` — table of what you tested and found
- Config entries in `agent/experiment_configs.py` — new experiment presets
- Any documentation changes

### Upload to Google Drive (large files)
- Checkpoint `.zip` files >10MB — created by training runs
- TensorBoard log directories — debugging/tuning artifacts
- Final trained model for competition submission

After uploading a checkpoint, add a row to `checkpoints/README.md`:

```markdown
| Member | Branch | Config | Win Rate | Drive Link |
|--------|--------|--------|----------|------------|
| YourName | team/arch-variants | dueling_heatmap | 68% | [link](https://drive.google.com/...) |
```

---

## 📝 Recording Results

After each experiment, update `RESULTS.md` with your findings. Use this format:

```markdown
### Area: [Your Area] — [Member Name]

| Experiment | Config | Steps | Win Rate vs no_op | Win Rate vs random | Win Rate vs square | Notes |
|-----------|--------|-------|-------------------|-------------------|-------------------|-------|
| Baseline | stack_16_heatmap | 50k | 100% | 88% | 72% | Strong baseline |
| Variant A | stack_16_heatmap_ent0.05 | 50k | 100% | 91% | 78% | Higher entropy helped |
```

This table is what the team uses to decide which configs go into the **final combined training run**.

---

## 📊 How to Run Experiments

### Quick test (10k steps, 2 envs)
```bash
python run_experiment.py --config heatmap_obs --steps 10000 --envs 2
```

### Full training (300k steps, 8 envs)
```bash
python agent/train_selfplay_v2.py
```

### Evaluate a trained checkpoint
```bash
python -c "
from agent.eval_harness import MatchEvaluator
from agent.wrapper_vec import make_unity_env_fn
from stable_baselines3 import PPO

model = PPO.load('path/to/checkpoint.zip')
env_fn = make_unity_env_fn(rank=0, build_path='path/to/Build.app')
evaluator = MatchEvaluator(model, env_fn)
results = evaluator.benchmark_all(n_matches=50)
print(results)
"
```

### Hyperparameter sweep
```bash
python scripts/grid_search.py sweep_configs/phase3_focused.json --output results/sweeps
```

---

## 📋 Training Plan

A 4-phase workflow for running, selecting, and combining experiments.

```
Trial experiments ──▶ Selection ──▶ Combination test ──▶ Final training
  10k–20k steps         no training       50k steps            300k steps
```

### Training Length Comparison

| Phase | Purpose | Steps (Proposed) | Our Setup | What This Means |
|-------|---------|-----------------:|----------:|-----------------|
| **Trial experiments** | Quickly test whether each idea helps | 10k–30k | 10k–20k | Short runs to filter ideas. Use `run_experiment.py --steps 10000 --envs 2`. Sweeps run 20k/trial. |
| **Selection** | Compare which config performs better | No training | No training | Evaluate checkpoints from trials. 50 matches each side, record win rates. |
| **Combination test** | Test selected best parts together | 30k–50k | 50k minimum | Combine best variant with baseline, run 50k. This is the **Done Criteria** check — evaluate both sides, 85% vs square = passed. |
| **Final training** | Train the final selected model | 100k+ | **300k** (100k × 3 stages) | 3-stage curriculum: 30% no-op → 30% square → 40% self-play. Run by M5 once all checkpoints are in. |

### Why 300k?

The curriculum makes every step count:

| Stage | Opponent | Steps | Goal |
|-------|----------|------:|------|
| 1 (30%) | No-op (stand still) | ~90k | Basic paddle control, ball tracking |
| 2 (30%) | Square-motion (predictable) | ~90k | Timing, positioning, shot selection |
| 3 (40%) | Self-play (past checkpoints) | ~120k | Strategy, adaptation, exploitation |

300k is **headroom**, not overkill — the agent isn't wasting early episodes on hard opponents. If it converges earlier we can stop early.

### Overfitting / Underfitting Risk

| Phase | Risk | Why |
|-------|------|-----|
| Trials (10–20k) | **None** | Too short to converge or memorize. Binary litmus test. |
| Combination (50k) | **Low** | Fixed opponent prevents overfitting. Enough for measurable skill. |
| Final (300k) | **Very low** | Self-play buffer (5 checkpoints) provides opponent diversity — the main regularization. |

### Per-Member Timeline

```
M3 (Observation)
└── Baseline (heatmap_obs, stack_16) ─────────▶ Everyone depends on this
                                                    │
                         ┌──────────────────────────┼──────────────────────────┐
                         ▼                          ▼                          ▼
                  M1 (Architecture)          M2 (Reward)              M4 (Hyperparams)
                  Trials → Sel → Combo      Trials → Sel → Combo     Trials → Sel
                         │                          │                          │
                         └──────────────────────────┼──────────────────────────┘
                                                    ▼
                                           M5 (Training Strategy)
                                           Stage 1+2 → Stage 3 (self-play)
                                           ──▶ Final model
```

### Key Rules

1. **M3 produces the observation baseline first** — everyone else depends on it.
2. **Each member runs their own trials + selection + combination test** independently.
3. **M5 can start Stage 1+2** before receiving checkpoints from others.
4. **M5 only starts Stage 3** after receiving at least 1 checkpoint from M1, M2, and M4.
5. **Checkpoints go to Google Drive** — log the link in your member file's experiment log.

---

## 🤝 Pull Request & Merge Process

**Step 1:** When your experiment is complete and your code is ready, open a Pull Request on GitHub:

```
Branch: team/YOUR-BRANCH → base: main
Title:  "[Your Area] Brief summary of changes"
Body:   What you changed, what results you got, what the team should know
```

**Step 2:** Keven reviews the PR:
- Checks that only your assigned files are modified
- Reviews logic for correctness
- Adds any comments/questions

**Step 3:** Keven clicks **"Squash and merge"** — all your commits become one clean commit on main.

**Step 4:** Once all 4 areas are merged, **one person** (Keven) does the final combined training run:

```bash
git checkout main
git pull
# Combine winning configs → train 300k steps → evaluate → package for competition
```

---

## ⚠️ Important Rules

1. **Only modify files in your assigned area.** If you need a shared file change, PR it first.
2. **Never push to `main` directly.** All changes go through Pull Requests.
3. **Pull latest main before starting work** to avoid merge conflicts.
4. **Don't commit large files** (`.zip`, `.pt`, `.pkl`, `.png`, TensorBoard logs). Use Google Drive.
5. **Keep `RESULTS.md` updated** — it's the team's decision-making document.
6. **Ask questions** — if anything is unclear about the code, your area, or git, ask in the group chat.

---

## ❓ Git Cheat Sheet

```bash
# See your current branch
git branch

# See all branches (local + remote)
git branch -a

# Switch to another branch
git checkout team/reward-shaping

# See what's changed but not yet staged
git diff

# Undo git add for a file
git reset HEAD filename.py

# Abandon local changes and reset to latest from remote
git checkout -- filename.py

# See teammate's branch (after they push)
git fetch origin
git checkout team/obs-variants

# Tag final submission
git tag -a v1.0 -m "Competition submission"
git push origin v1.0
```
