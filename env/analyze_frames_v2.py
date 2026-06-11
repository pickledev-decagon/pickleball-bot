import cv2
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FRAMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captured_frames")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

frames = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith('.png')])
print(f"Found {len(frames)} frames")

# Collect color stats across all frames
h_all, s_all, v_all = [], [], []
ball_positions = []

# Test multiple HSV ranges
test_ranges = [
    ("yellow (narrow)", np.array([20, 80, 80]), np.array([35, 255, 255])),
    ("yellow (wide)", np.array([15, 50, 80]), np.array([45, 255, 255])),
    ("white", np.array([0, 0, 200]), np.array([180, 30, 255])),
    ("bright (any color)", np.array([0, 0, 150]), np.array([180, 80, 255])),
]

for fname in frames:
    path = os.path.join(FRAMES_DIR, fname)
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        continue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    h_all.append(hsv[:,:,0].flatten())
    s_all.append(hsv[:,:,1].flatten())
    v_all.append(hsv[:,:,2].flatten())

print(f"\nOverall HSV statistics across {len(frames)} frames:")
h_flat = np.concatenate(h_all)
s_flat = np.concatenate(s_all)
v_flat = np.concatenate(v_all)
for name, data in [("Hue", h_flat), ("Sat", s_flat), ("Val", v_flat)]:
    print(f"  {name}: min={data.min()}, max={data.max()}, mean={data.mean():.1f}, median={np.median(data):.1f}")

print(f"\nBall detection results per range:")
for range_name, lower, upper in test_ranges:
    from env.preprocessing import BallTracker
    tracker = BallTracker()
    tracker.set_hsv_range(lower, upper)
    detections = 0
    for fname in frames[:20]:
        path = os.path.join(FRAMES_DIR, fname)
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        bx, by = tracker.detect(img_rgb)
        if bx >= 0:
            detections += 1
    pct = detections / min(len(frames), 20) * 100
    print(f"  {range_name}: {detections}/{min(len(frames), 20)} detected ({pct:.0f}%)")

# For best range, save visualizations
best_idx = 0  # first range is default
tracker = BallTracker()
for range_name, lower, upper in test_ranges:
    print(f"\n{range_name}: T={lower.tolist()} to {upper.tolist()}")

print("\nDone. Check analysis_output/ for visual results.")
print("\nTo view images: open env/analysis_output/")
