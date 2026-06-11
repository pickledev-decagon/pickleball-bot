"""
Central project configuration.

All paths, hyperparameters, and cross-file constants live here.
Import this from any script that needs the Unity build path, training
directories, or environment defaults.

To override the Unity build path without editing this file, set the
environment variable DPICKLEBALL_BUILD_PATH:

    export DPICKLEBALL_BUILD_PATH="/path/to/your/build.app"

Or create a .env file in the project root:

    DPICKLEBALL_BUILD_PATH=/path/to/your/build.app
"""

import os
import pathlib

# --- Project root ---
PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()

# --- Unity build path ---
# Precedence: env var > .env file > default (Mac)
_BUILD_PATH = os.environ.get("DPICKLEBALL_BUILD_PATH", "")

if not _BUILD_PATH:
    # Try loading from .env
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "DPICKLEBALL_BUILD_PATH":
                    _BUILD_PATH = v.strip().strip("\"'")
                    break

if not _BUILD_PATH:
    # Default for Mac (most common dev environment)
    _BUILD_PATH = os.path.expanduser(
        "~/Downloads/dPickleball BuildFiles/Training/Mac/Mac.app"
    )

BUILD_PATH = _BUILD_PATH


# --- Training directories ---
CHECKPOINT_DIR = str(PROJECT_ROOT / "checkpoints_selfplay_v2")
LOG_DIR = str(PROJECT_ROOT / "logs_selfplay_v2")
BUFFER_DIR = str(PROJECT_ROOT / "opponent_buffer_v2")
RESULTS_DIR = str(PROJECT_ROOT / "results")


# --- Environment defaults ---
NUM_ENVS = 8
FRAME_STACK = 16


# --- Curriculum stages ---
STAGE1_FRAC = 0.30  # no-op opponent + dense reward
STAGE2_FRAC = 0.30  # square-motion opponent + fading reward
STAGE3_FRAC = 0.40  # self-play from buffer + sparse only
TIMESTEPS_PER_STAGE = 100_000
TOTAL_TIMESTEPS = int(TIMESTEPS_PER_STAGE * 3)


# --- PPO defaults ---
PPO_DEFAULTS = {
    "learning_rate": 2.5e-4,
    "n_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}


# --- Opponent buffer ---
BUFFER_MAX_SIZE = 5


def check_build_path():
    """Verify the build path exists. Returns True if valid."""
    exists = os.path.exists(BUILD_PATH)
    if not exists:
        print(f"⚠️  Unity build not found at: {BUILD_PATH}")
        print("   Set DPICKLEBALL_BUILD_PATH env var or create .env file.")
    return exists
