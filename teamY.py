"""Competition entry point — left-side agent (teamY, with horizontal mirror)."""

import sys, os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent.competition_export import TeamLeft

# Auto-discover latest trained checkpoint
BUFFER_DIR = Path(__file__).parent / "opponent_buffer_v2"
CKPTS = sorted(BUFFER_DIR.glob("curriculum_final_*.zip"))
if not CKPTS:
    raise FileNotFoundError(
        f"No curriculum_final_*.zip found in {BUFFER_DIR.resolve()}. "
        "Run train_selfplay_v2.py first."
    )
FINAL_MODEL = str(CKPTS[-1])


def create_agent():
    return TeamLeft(
        FINAL_MODEL,
        device="cpu",
        frame_stack=16,
        use_heatmap=True,
        heatmap_sigma=5,
        heatmap_weight=0.7,
    )
