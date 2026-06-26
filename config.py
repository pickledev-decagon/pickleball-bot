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
from dataclasses import dataclass

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
STAGE2_FRAC = 0.70  # square-motion opponent + fading reward
STAGE3_FRAC = 0.00  # self-play — REMOVED (exploits past selves, doesn't transfer)
TIMESTEPS_PER_STAGE = 100_000
TOTAL_TIMESTEPS = int(TIMESTEPS_PER_STAGE * 2)  # 200k, 2 stages only


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


# --- Reproducibility ---
SEED = 42


# --- Penalty defaults ---
ROTATION_PENALTY_DEFAULT = 0.001


# === Observation pipeline → reward parameter coupling ===
# The camera assignment determines where the paddle appears in the
# processed observation. Reward proximity must reference the correct
# x-coordinate. Change this if you modify env/wrapper.py's camera logic.

@dataclass
class ObservationConfig:
    """Describes the observation pipeline for reward consistency validation.

    When the camera assignment changes, the effective paddle position in the
    processed view changes. This config ensures the reward function stays in
    sync with the observation pipeline.

    - per_agent: each agent reads its own camera → paddle at x≈0.65
    - shared:   both agents read P0's camera (old bug) → paddle at x≈0.35

    If you change camera_assignment here, the effective_paddle_x is derived
    automatically. If you provide a mismatched pair, __post_init__ raises.
    """
    camera_assignment: str = "per_agent"
    effective_paddle_x: float = 0.65

    def __post_init__(self):
        CAMERA_CONSISTENCY = {
            "per_agent": 0.65,
            "shared": 0.35,
        }
        expected = CAMERA_CONSISTENCY.get(self.camera_assignment)
        if expected is None:
            raise ValueError(
                f"Unknown camera_assignment={self.camera_assignment!r}. "
                f"Choose from: {list(CAMERA_CONSISTENCY.keys())}"
            )
        if abs(self.effective_paddle_x - expected) > 0.01:
            raise ValueError(
                f"Effective paddle_x={self.effective_paddle_x} is "
                f"inconsistent with camera_assignment={self.camera_assignment!r} "
                f"(expected ~{expected}). Update effective_paddle_x to match, "
                f"or change camera_assignment."
            )

OBSERVATION_CONFIG = ObservationConfig()
# Module-level singleton. Import and use wherever paddle_x is needed.


def check_build_path():
    """Verify the build path exists. Returns True if valid."""
    exists = os.path.exists(BUILD_PATH)
    if not exists:
        print(f"⚠️  Unity build not found at: {BUILD_PATH}")
        print("   Set DPICKLEBALL_BUILD_PATH env var or create .env file.")
    return exists
