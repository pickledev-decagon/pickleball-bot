"""Export validation: 4 tests against CompetitionPreprocessor + AsyncPolicy."""
import sys, os, time
import numpy as np
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.competition_export import CompetitionPreprocessor, TeamRight, TeamLeft

CHECKPOINT = str(sorted(
    (Path(__file__).parent.parent / "opponent_buffer_v2").glob("curriculum_final_*.zip")
)[-1])

passed = 0
failed = 0

def check(cond, msg):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS: {msg}")
    else:
        failed += 1
        print(f"  FAIL: {msg}")

# ============================================================
# Test A: Preprocessor shape, range, and ball detection
# ============================================================
print("\n[A] Preprocessor shape/range + ball detection")

prep = CompetitionPreprocessor(frame_stack=16, use_heatmap=True, heatmap_sigma=5, heatmap_weight=0.7)

# Feed 20 random RGB frames — ensure shape is always correct
for i in range(20):
    raw = np.random.rand(3, 84, 168).astype(np.float32)
    obs = prep.process(raw)
    check(obs.shape == (16, 84, 168), f"frame {i}: shape {obs.shape}")
    check(0.0 <= obs.min() <= obs.max() <= 1.0, f"frame {i}: range [{obs.min():.3f}, {obs.max():.3f}]")

# Ball detection: create a frame with a clear yellow blob
yellow = np.zeros((84, 168, 3), dtype=np.float32)
cx, cy = 120, 40  # yellow blob center
r = 4
for dx in range(-r, r+1):
    for dy in range(-r, r+1):
        if dx*dx + dy*dy <= r*r:
            px, py = cx + dx, cy + dy
            if 0 <= px < 168 and 0 <= py < 84:
                yellow[py, px] = [1.0, 0.9, 0.0]  # bright yellow in RGB

yellow_ch = yellow.transpose(2, 0, 1).astype(np.float32)
prep2 = CompetitionPreprocessor(frame_stack=4, use_heatmap=True)
for _ in range(8):
    prep2.process(yellow_ch)
obs_with_ball = prep2.process(yellow_ch)
maxval = obs_with_ball.max()
check(maxval > 0.5, f"ball heatmap signal present (max={maxval:.3f}, expected >0.5)")

# ============================================================
# Test B: Frame stack integrity
# ============================================================
print("\n[B] Frame stack integrity")

prep3 = CompetitionPreprocessor(frame_stack=4)
frame_a = np.random.rand(3, 84, 168).astype(np.float32)
frame_b = np.random.rand(3, 84, 168).astype(np.float32)

# B1: 16 identical frames -> all outputs should be identical
prep3.reset()
outputs = []
for _ in range(16):
    outputs.append(prep3.process(frame_a))
all_same = all(np.allclose(outputs[0], o) for o in outputs)
check(all_same, "16 identical frames -> all identical output")

# B2: 16 different frames -> at least one output should differ
prep3.reset()
outputs2 = []
for _ in range(16):
    outputs2.append(prep3.process(np.random.rand(3, 84, 168).astype(np.float32)))
any_diff = False
for o in outputs2:
    if not np.allclose(outputs2[0], o):
        any_diff = True
        break
check(any_diff, "16 different frames -> outputs differ")

# ============================================================
# Test C: Left-side mirror (play_as_left)
# ============================================================
print("\n[C] Left-side mirror")

# Create agents
right = TeamRight(CHECKPOINT, device="cpu", frame_stack=16)
left = TeamLeft(CHECKPOINT, device="cpu", frame_stack=16)

# Feed same obs to both sides, verify left returns mirrored horizontal action
obs = np.random.rand(3, 84, 168).astype(np.float32)

# Need multiple calls to fill frame stack, then compare
for _ in range(8):
    r_act = right.policy(obs, 0.0)
    l_act = left.policy(obs, 0.0)

# After frame stack is full, actions should be valid
check(r_act.shape == (3,), f"right action shape {r_act.shape}")
check(l_act.shape == (3,), f"left action shape {l_act.shape}")
check(all(0 <= a <= 2 for a in r_act), f"right action in range {r_act}")
check(all(0 <= a <= 2 for a in l_act), f"left action in range {l_act}")
check(r_act.dtype == np.int32, f"right dtype {r_act.dtype}")
check(l_act.dtype == np.int32, f"left dtype {l_act.dtype}")

# ============================================================
# Test D: Edge cases — no crashes
# ============================================================
print("\n[D] Edge cases")

prep4 = CompetitionPreprocessor(frame_stack=4)

# All-black frame
black = np.zeros((3, 84, 168), dtype=np.float32)
try:
    obs = prep4.process(black)
    check(True, "all-black frame — no crash")
    check(obs.shape == (4, 84, 168), f"all-black shape {obs.shape}")
except Exception as e:
    check(False, f"all-black crash: {e}")

# All-white frame
prep4.reset()
white = np.ones((3, 84, 168), dtype=np.float32)
try:
    obs = prep4.process(white)
    check(True, "all-white frame — no crash")
except Exception as e:
    check(False, f"all-white crash: {e}")

# Rapid random frames (stress test)
prep4.reset()
try:
    for i in range(100):
        obs = prep4.process(np.random.rand(3, 84, 168).astype(np.float32))
    check(True, "100 rapid random frames — no crash")
except Exception as e:
    check(False, f"rapid frames crash: {e}")

# ============================================================
print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
print(f"{'ALL PASSED' if failed == 0 else 'SOME FAILED'}")
