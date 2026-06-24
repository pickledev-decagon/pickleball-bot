"""Quick diagnostic: print each agent's observation space from the Unity build.

Usage:
    python scripts/check_obs_spaces.py

Shows which agents have cameras (≥2D observation) and which only have vectors.
"""
import sys
sys.path.insert(0, ".")

from agent.wrapper_vec import make_unity_env_fn
from config import BUILD_PATH

env_fn = make_unity_env_fn(rank=0, build_path=BUILD_PATH, frame_stack=16)
env = env_fn()

print("=== Observation Spaces ===")
for aid in env.env.possible_agents:
    space = env.env.observation_spaces[aid]
    shape_str = str(space)
    has_camera = False
    if hasattr(space, "spaces"):  # Tuple
        for s in space.spaces:
            if len(s.shape) >= 2:
                has_camera = True
    elif len(space.shape) >= 2:
        has_camera = True
    print(f"  {aid}: {shape_str}  {'✅ has camera' if has_camera else '❌ vector only'}")

print()
for aid in env.env.possible_agents:
    space = env.env.observation_spaces[aid]
    if hasattr(space, "spaces"):
        for i, s in enumerate(space.spaces):
            print(f"  {aid} sub-space [{i}]: shape={s.shape}, dtype={s.dtype}")
    else:
        print(f"  {aid} single space: shape={space.shape}, dtype={space.dtype}")

env.close()
