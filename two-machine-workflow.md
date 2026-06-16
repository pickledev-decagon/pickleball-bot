# Two-Machine Workflow Plan

**Date:** June 16, 2026
**Context:** M5 MacBook Air (development, git, coordination) + M1 MacBook Pro (24/7 training)

---

## 1. Overview

```
┌─────────────────────────────────┐          ┌─────────────────────────────────┐
│      M5 MacBook Air (You)       │          │    M1 MacBook Pro (Leftover)    │
│                                 │          │                                 │
│  Role: Development + Git + PR   │          │  Role: Heavy Training Only      │
│                                 │          │                                 │
│  Hermes: This session           │          │  Hermes: 24/7 separate instance │
│                                 │          │                                 │
│  └─ Write code                  │          │  └─ git pull (read-only)        │
│  └─ git commit & push           │          │  └─ conda env + Unity build     │
│  └─ Manage branches, PRs        │          │  └─ Run training experiments    │
│  └─ Review, merge               │  GitHub   │  └─ Upload checkpoints → Drive  │
│  └─ Coordination docs           │═══╦═══    │  └─ Log results in progress/   │
│                                 │   ║       │                                 │
│  Google Drive (checkpoints) ◄───╬───╬───────►  Google Drive (uploads)        │
└─────────────────────────────────┘   ║       └─────────────────────────────────┘
                                      ║
                              pickledev-decagon/
                              pickleball-bot.git
```

### Key Principle

> **M5 is the Source of Truth. M1 only reads from GitHub, never pushes.**

All code changes, documentation, and git history originate on M5. The M1 clone exists solely to run `git pull` and execute training scripts. The M1's Hermes agent does not have write access to the repo.

---

## 2. M1 Initial Setup (One-Time)

These steps are done **once** on the M1 to make it training-ready. After this, the M1 only needs `git pull` before each training run.

### Step 2.1 — Clone the repo

```bash
cd ~/Documents
git clone https://github.com/pickledev-decagon/pickleball-bot.git
cd pickleball-bot
git submodule update --init --depth 1
```

### Step 2.2 — Create conda environment

```bash
conda create -n dpickleball pip python=3.10.12 -y
conda activate dpickleball

pip install -e ./dpickleball-ml-agents/ml-agents-envs
pip install stable-baselines3 opencv-python==4.7.0.72 \
            matplotlib numpy "torch>=2.3,<2.6" "torchvision>=0.18,<0.20"

# If numpy issues:
pip install "numpy<2,>=1.23.5"
pip install "setuptools<70"
```

### Step 2.3 — Download Unity build and configure `.env`

1. Download `dPickleball BuildFiless.zip` (395 MB) from [Google Drive](https://drive.google.com/drive/folders/1lFqj6lopoIO96C_IO8yGXGzRMY1S4fzi)
2. Unzip to a stable location (e.g., `~/Documents/dp_build/`)
3. Set up `.env`:

```bash
echo 'DPICKLEBALL_BUILD_PATH="/Users/keven/Documents/dp_build/Training/Mac/Mac.app"' > .env
chmod -R 755 /Users/keven/Documents/dp_build/Training/Mac/Mac.app
```

> **Note:** The M1 path is `/Users/keven/Documents/dp_build/Training/Mac/Mac.app`. The M5 path is whatever is set in its own `.env`. Both download the same Unity build from Google Drive.

### Step 2.4 — Verify

```bash
python run_experiment.py --list
python run_experiment.py --config baseline --steps 2000 --envs 2
```

### Step 2.5 — Cleanup alias (add to `~/.zshrc`)

```bash
alias killdp='pkill -f "Mac.app/Contents/MacOS/dp"'
```

---

## 3. Daily Workflow

### On M5 (Development + Git)

```
  ┌─────────────────────────────┐
  │  Write/Edit code on M5      │
  │  git add, commit, push      │
  │  (to team/* branches → PR)  │
  └──────────┬──────────────────┘
             │ Code on GitHub
             ▼
  ┌─────────────────────────────┐
  │  Merge PRs to main          │
  │  (Keven reviews + squash)   │
  └──────────┬──────────────────┘
             │ main updated
             ▼
  ┌─────────────────────────────┐
  │  Document in README, docs   │
  │  Update DASHBOARD.md, etc.  │
  └─────────────────────────────┘
```

### On M1 (Training Only — launched via Hermes or terminal)

```
  ┌─────────────────────────────┐
  │  git pull origin main       │
  │  (or switch to a team/*     │
  │   branch for testing)       │
  └──────────┬──────────────────┘
             │ Fresh code
             ▼
  ┌─────────────────────────────┐
  │  Run training experiment    │
  │  python run_experiment.py   │
  │  or python agent/train_... │
  └──────────┬──────────────────┘
             │ Checkpoints + results
             ▼
  ┌─────────────────────────────┐
  │  Upload checkpoint → Drive  │
  │  Log results to Google Doc  │
  │  (or push result JSON to    │
  │   a data-only branch)       │
  └─────────────────────────────┘
```

---

## 4. Data Flow Between Machines

| Data Type | Direction | Method | Notes |
|-----------|-----------|--------|-------|
| Code changes | M5 → GitHub → M1 | `git push` → `git pull` | M1 never pushes |
| Experiment configs | M5 → GitHub → M1 | Same as code | M5 defines configs in `sweep_configs/` |
| Training checkpoints | M1 → Google Drive | Manual upload | Log link in progress file |
| Experiment results (JSON) | M1 → M5 | `data/results` branch + Drive | M1 commits JSONs to `data/results` — see §4 |
| Unity build | M1 downloads from Drive | Same URL for all | Separate per-machine download |
| `.env` | Machine-local | Created once per machine | Different paths per Mac |

### Combined Approach: GitHub `data/results` + Google Drive

**Hybrid flow:**

| Data | Method | Why |
|------|--------|-----|
| **Result JSONs** (`results/*.json`) | M1 commits to `data/results` branch | Small files, stay in-repo, easy for M5 to pull and incorporate into `RESULTS.md` |
| **Checkpoints** (`.zip`, `.pt`, `.pkl`) | Upload to Google Drive | Too large for git |
| **TensorBoard logs** | Upload to Google Drive | Too large for git |

**M1 Hermes workflow after each experiment:**

```bash
# 1. Commit result JSON to data/results branch
git add results/*.json
git commit -m "results: baseline_heatmap 50k — win rate 72% vs square"
git push origin HEAD:data/results

# 2. Upload checkpoints to Drive (use gdrive CLI or browser)
#    https://drive.google.com/drive/folders/1Mi7xp03CszctgAkf6dkiUdYPrH6Z5ibz

# 3. Switch back to main / training branch
git checkout main
```

**M5 workflow to collect results:**

```bash
git fetch origin
git checkout data/results
git pull
# copy result JSONs to local results/
# update RESULTS.md and tracking docs
git checkout main
```

The `data/results` branch has no code — only JSON files and possibly a log of what was tested. No PR needed, no review. It's purely a data pipeline.

> **⏳ This branch doesn't exist yet.** It must be created first (Step 1 in implementation order). Create it from `main` with no files:
> ```bash
> git checkout --orphan data/results
> git reset --hard
> git commit --allow-empty -m "init: data/results branch for experiment result JSONs"
> git push origin data/results
> git checkout main
> ```

---

## 5. Branch Strategy with Two Machines

### Standard Workflow

```
M5 creates:          team/obs-variants          (code changes)
M5 push → GitHub     team/obs-variants
                     ↓ Open PR
M5 merge → main
                     ↓
M1 pull →            main                       (read-only clone)
```

### What If M1 Needs to Test Unmerged Code?

**Scenario 1:** You (on M5) push changes to `team/hyperparams` and want M1 to test the branch before merging to `main`.

On M1:
```bash
cd ~/Documents/pickleball-bot

# Fetch all branches from GitHub
git fetch origin

# Check out the team branch (M1 tracks it as a detached local branch)
git checkout -b team/hyperparams origin/team/hyperparams

# Run experiments on this code...
python run_experiment.py --config entropy_0.05 --steps 20000 --envs 8

# Results will be in results/ — commit to data/results branch
git checkout main  # switch off the team branch first
git add results/*.json
git commit -m "results: entropy_0.05 from team/hyperparams"
git push origin HEAD:data/results
```

**Scenario 2:** A teammate (Amber, Yenly, etc.) pushes their branch and wants it tested.

On M1:
```bash
git fetch origin
git checkout -b team/arch-variants origin/team/arch-variants
# ... run experiments ...
git checkout main  # always switch back before committing results
```

**Important rules for M1:**
- ❌ Never push to someone else's `team/*` branch
- ❌ Never push to `main`
- ✅ Push only to `data/results` (JSON files only)
- ✅ Always `git checkout main` before committing results — this keeps result commits clean and separate from experiment code
- If you need to test multiple branches, just `git checkout -b` each one, the local branch disappears when deleted (the remote copy is the source of truth)

### Branch Access Table

| Branch | M5 (Dev) | M1 (Training) |
|--------|----------|---------------|
| `main` | Push (via PR) | `git pull` (read-only) |
| `team/*` | Push directly | `git fetch` + `git checkout -b` (read-only) |
| `data/results` | `git pull` (collect results) | Push result JSONs only |

---

## 6. M1 Hermes Agent Setup (For the M1 Instance)

The M1's own Hermes agent (separate instance, separate memory) should be configured with:

### Skills to Load
- **`pickleball-bot-training`** — dedicated skill (create on M1's Hermes) that defines:
  1. `cd ~/Documents/pickleball-bot`
  2. `git pull origin main`
  3. `conda activate dpickleball`
  4. Run the targeted experiment
  5. Upload checkpoint to Google Drive

### Example M1 Hermes Prompt

```
Load the pickleball-bot-training skill.
Pull latest main, then run:
  python run_experiment.py --config heatmap_obs --steps 10000 --envs 8
When done, upload any new checkpoint .zip files to the shared Drive folder.
```

### Cron Jobs on M1 (Optional)

The M1 can run **nightly training cycles** via Hermes cron:

| Cron Schedule | Task |
|-------------|------|
| Daily at 2 AM | Pull main → run a sweep config → upload checkpoints |
| After sweep | Run eval harness on best checkpoint |

---

## 7. Checkpoint Management

Since M1 and M5 do not share a filesystem:

1. **M1 generates checkpoints** during training (saved to `checkpoints_selfplay_v2/`)
2. **M1 uploads to Google Drive** (the shared folder)
3. **M5 or anyone** can download from Drive for evaluation or further training

**Critical: Do not store checkpoints in git.** The `.gitignore` already excludes `.zip`, `*.pt`, `*.pkl`, and checkpoint directories.

### Upload Checklist (for M1 Hermes prompt)

After training, the M1 Hermes agent should:

1. List `.zip` files in `checkpoints_selfplay_v2/` or `opponent_buffer_v2/`
2. Upload each to [shared Drive folder](https://drive.google.com/drive/folders/1Mi7xp03CszctgAkf6dkiUdYPrH6Z5ibz)
3. Log the Drive link in the progress file or a Google Doc
4. Notify the team (or write to `RESULTS.md`-equivalent doc on Drive)

---

## 8. Hermes Agent Memory Separation

### M5 Hermes (This Instance)
- Full project context: team members, branches, code structure
- Focus: development, git management, PR review, documentation
- Memory: Keven is lead/merger, team branches, Shared Drive links

### M1 Hermes (24/7 Instance)
- Limited project context: training-only workflows
- Focus: pull latest code, run experiments, upload checkpoints
- Memory: repo path, conda env name, Drive folder ID, Unity build path
- Should **not** track development plans, team members, or PR status — that's M5's domain

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| M1 tries to push to main | M1 clone configured with read-only token or no push URL |
| M1 runs stale code | Always `git pull` before training — script this as the first step |
| Checkpoints lost on M1 without Drive backup | Hermes prompt should upload after each run |
| Unity build version mismatch | Both machines download same build from Drive; update via Drive |
| M1 Hermes memory drifts from M5's | Keep M1 Hermes memory focused on training-only facts; M5 owns the full context |
| M1 runs out of disk (checkpoints pile up) | Periodically: `rm -rf checkpoints_selfplay_v2/ opponent_buffer_v2/ logs_selfplay_v2/` (after Drive upload) |

---

## 10. Recommended Implementation Order

| Step | What | Who | Est. Time |
|------|------|-----|-----------|
| 1 | **Create `data/results` branch** on GitHub (empty, no code) | M5 | 1 min |
| 2 | **Write this plan** → commit to repo on M5 | M5 | 15 min |
| 3 | Set up M1: clone repo, conda env, Unity build | M1 | 20 min |
| 4 | Verify M1 can run a short experiment (2k steps) | M1 | 5 min |
| 5 | Set up Google Drive upload workflow on M1 | M1 | 10 min |
| 6 | Create `pickleball-bot-training` skill on M1 Hermes | M1 | 10 min |
| 7 | Run first real experiment on M1 | M1 | depends |
| 8 | Collect results on M5 and update tracking docs | M5 | 10 min/cycle |
