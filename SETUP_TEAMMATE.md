# Teammate Setup Guide

5 steps to get your machine ready to run experiments. Estimated time: 10-15 minutes.

---

## Step 1: Create Conda Environment

```bash
conda create -n dpickleball pip python=3.10.12 -y
conda activate dpickleball
```

## Step 2: Install ML-Agents Fork

```bash
git clone https://github.com/dPickleball/dpickleball-ml-agents.git
cd dpickleball-ml-agents
pip install -e ./ml-agents-envs
pip install -e ./ml-agents
cd ..
```

## Step 3: Install Dependencies

```bash
pip install stable-baselines3 opencv-python==4.7.0.72 \
            matplotlib numpy "torch>=2.3,<2.6" "torchvision>=0.18,<0.20"
```

If you get numpy version errors:
```bash
pip install "numpy<2,>=1.23.5"
pip install "setuptools<70"
```

## Step 4: Download Unity Build

1. Go to: https://drive.google.com/drive/folders/1lFqj6lopoIO96C_IO8yGXGzRMY1S4fzi
2. Download `dPickleball BuildFiless.zip` (395 MB)
3. Unzip to e.g. `~/Desktop/dp_build/`
4. Check the path contains `Training/Mac/Mac.app/Contents/MacOS/dp`

```bash
chmod -R 755 ~/Desktop/dp_build/Training/Mac/Mac.app
```

## Step 5: Get the Code & Run

```bash
cd pickleball_bot

# Edit BUILD_PATH in run_experiment.py to match your Unity build location:
#   line 25: BUILD_PATH = "/path/to/your/Training/Mac/Mac.app"

# Verify everything is set up:
python run_experiment.py --list

# Quick test (2 min, 2 parallel envs):
python run_experiment.py --config baseline --steps 2000 --envs 2
```

---

## `run_experiment.py` CLI Reference

```bash
python run_experiment.py <flags>
```

| Flag | Default | Description |
|---|---|---|
| `--config <name>` | `baseline` | Experiment preset. See `--list` for options. |
| `--steps <N>` | `10000` | Total timesteps across all envs |
| `--envs <N>` | `2` | Number of parallel Unity environments |
| `--list` | — | Print available configs and exit |

**Examples:**
```bash
# M1 (2 envs, short test)
python run_experiment.py --config entropy_0.05 --steps 5000 --envs 2

# M5 Pro (8 envs, standard run)
python run_experiment.py --config entropy_0.05 --steps 10000 --envs 8

# Quick multi-config sweep
for cfg in baseline entropy_0.05 rgb_obs; do
  python run_experiment.py --config $cfg --steps 10000 --envs 4
done
```

---

## Output Format

Each run produces a JSON file in `results/`:

```json
results/baseline_20260513_163541.json
```

```json
{
  "experiment": "baseline",
  "description": "Current default: grayscale, 4-stack, entropy 0.01, curriculum, proximity reward",
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

**After each run:** Copy the JSON content into [RESULTS.md](RESULTS.md) under the corresponding experiment section. Update the Quick Comparison table.

---

## Cleanup Instructions

If experiments crash or you need to stop, you may have leftover Unity processes:

```bash
# Kill all Unity processes
pkill -f "Mac.app/Contents/MacOS/dp"

# Verify they're gone
ps aux | grep dp | grep -v grep

# Free up ports (5005-5012 range)
lsof -ti:5005-5012 | xargs kill -9 2>/dev/null || true
```

Add this alias to `~/.zshrc`:
```bash
alias killdp='pkill -f "Mac.app/Contents/MacOS/dp"'
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Permission denied` on Unity build | `chmod -R 755 /path/to/Mac.app` |
| `_ARRAY_API not found` | `pip install "numpy<2"` |
| `gym` / `gymnasium` conflicts | `pip install gymnasium && pip uninstall gym -y` |
| `No module named 'mlagents_envs'` | Did you run Step 2? Make sure dpickleball-ml-agents is cloned and pip-installed. |
| `Address already in use` | Unity port still occupied. Run cleanup commands above. |
| Python version wrong | `conda activate dpickleball && python --version` should show `3.10.x` |
| `tensorboard` not found | `pip install tensorboard` |
| Unity window flashing on/off | Normal — it's the gRPC connection establishing. Wait 3-5 seconds. |
| Very slow step time (>50ms) | Close other GPU-intensive apps. Reduce `--envs`. Check `htop` for memory pressure. |

---

## After Setup

You'll receive an experiment assignment (which configs to run). For each:

```bash
python run_experiment.py --config <name> --steps 20000 --envs 8
```

Each experiment takes ~10 minutes on M5 Pro (8 envs). Paste the output JSON into RESULTS.md when done.
