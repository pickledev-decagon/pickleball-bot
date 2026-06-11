# Contributing Guide — dPickleBall Bot

## 📋 Team Roles

| Member | Branch | Area | Files They Own |
|--------|--------|------|---------------|
| **Keven** (you) | `team/arch-variants` | Model Architecture | `agent/two_tower_cnn.py`, `env/wrapper.py` |
| **Member 2** | `team/reward-shaping` | Reward Functions | `agent/reward_shaping.py`, `agent/experiment_configs.py` |
| **Member 3** | `team/obs-variants` | Observation | `env/preprocessing.py` (BallTracker, heatmap, frame diff) |
| **Member 4** | `team/hyperparams` | Hyperparameters | `sweep_configs/*.json`, `scripts/grid_search.py` |
| **Member 5** | `team/training-strategy` | Training Strategy | `agent/train_selfplay_v2.py`, `agent/selfplay_callback.py`, `agent/buffer.py`, `config.py` |

**Rule:** Only modify files in your assigned area. If you need a change in a shared file, open a small PR targeting `main` first.

---

## 🖥️ Initial Setup (Once)

```bash
# 1. Clone the repo (with submodule)
git clone https://github.com/pickledev-decagon/pickleball-bot.git
cd pickleball-bot
git submodule update --init --depth 1

# 2. Install ML-Agents dependency
pip install -e ./dpickleball-ml-agents/ml-agents-envs

# 3. Create conda environment and install packages
conda create -n dpickleball pip python=3.10.12
conda activate dpickleball
pip install stable-baselines3 opencv-python==4.7.0.72 numpy==1.26.4 torch

# 4. Set your Unity build path
echo 'DPICKLEBALL_BUILD_PATH="/absolute/path/to/your/Unity/Build.app"' > .env
```

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
