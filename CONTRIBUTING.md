# Contributing to PickleBall Bot

## Setup

```bash
# 1. Clone the repo (with submodule)
git clone https://github.com/pickledev-decagon/pickleball-bot.git
cd pickleball-bot
git submodule update --init --depth 1

# 2. Install ML-Agents dependency (from submodule)
pip install -e ./dpickleball-ml-agents/ml-agents-envs

# 3. Install other dependencies
pip install stable-baselines3 opencv-python numpy torch torchvision

# 4. Set your Unity build path
export DPICKLEBALL_BUILD_PATH="/path/to/your/Build.app"
# Or create .env file:
echo 'DPICKLEBALL_BUILD_PATH="/path/to/your/Build.app"' > .env
```

## Your Branch

Each member owns one area:

| Member | Branch | Area | Files |
|--------|--------|------|-------|
| **Keven** | `team/arch-variants` | Architecture | `agent/two_tower_cnn.py`, `env/wrapper.py` |
| **Member 2** | `team/reward-shaping` | Reward functions | `agent/reward_shaping.py`, `agent/experiment_configs.py` |
| **Member 3** | `team/obs-variants` | Observation | `env/preprocessing.py` |
| **Member 4** | `team/hyperparams` | Hyperparameters | `sweep_configs/*.json`, `scripts/grid_search.py` |

## Daily Workflow

```bash
# Start your day: get latest main
git checkout main
git pull
git submodule update --init --depth 1   # update ml-agents submodule if changed
git checkout team/YOUR_BRANCH
git merge main   # bring latest main into your branch

# Work, commit, push
git add <your-files>
git commit -m "Describe what you changed"
git push -u origin team/YOUR_BRANCH

# Open a Pull Request on GitHub when your piece is ready
# Keven reviews and merges
```

## Important Rules

1. **Only modify files in your assigned area.** If you need a change in a shared file (e.g. `config.py`), open a small PR targeting main first.
2. **Never push to `main` directly.** All changes go through Pull Requests.
3. **Don't commit large files** (`.zip`, `.pt`, `.pkl`, TensorBoard logs, captured frames). Use Google Drive for checkpoints >10 MB.
4. **Pull latest main before starting work** to avoid merge conflicts.

## What NOT to Commit

| File/Folder | Why |
|-------------|-----|
| `*.zip`, `*.pt`, `*.pkl` | Checkpoints are too large — upload to Google Drive |
| `checkpoints_selfplay_v2/` | Generated during training |
| `logs_selfplay_v2/` | TensorBoard logs |
| `opponent_buffer_v2/` | Generated during training |
| `results/` | Experiment JSONs |
| `archive/` | Old broken versions |
| `phase-*/` | Sandbox/planning |

## Recording Your Results

After training, update `RESULTS.md` (or create a section if yours is the first entry):

```markdown
### Arch: TwoTower CNN [Member: Keven]
| Config | Win Rate | Steps | Notes |
|--------|----------|-------|-------|
| dueling_heatmap | 68% vs square | 50k | Best architecture variant |
```

Upload your best checkpoint to Google Drive and add a link to `checkpoints/README.md`:

```markdown
| Member | Branch | Config | Win Rate | Drive Link |
|--------|--------|--------|----------|------------|
| Keven | arch-variants | dueling_heatmap | 68% | [link](https://drive.google.com/...) |
```

## Merging

1. All members finish their experiments
2. PRs merged to `main`
3. One person checks out `main`, trains final combined model
4. Final checkpoint uploaded to Drive, submission packaged
5. Tag the release: `git tag -a v1.0 -m "Competition submission"`
