import cv2
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(BASE_DIR, "captured_frames")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

frames = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith('.png')])
print(f"Found {len(frames)} frames")

# Analyze a few frames
sample_frames = frames[::20]  # every 20th frame

for fname in sample_frames[:5]:
    path = os.path.join(FRAMES_DIR, fname)
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        continue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, w = img_rgb.shape[:2]
    print(f"\n--- {fname} (size={w}x{h}) ---")

    # Analyze center region (where ball might be)
    # Sample: 4 quadrants
    regions = {
        "top-left": hsv[:h//2, :w//2],
        "top-right": hsv[:h//2, w//2:],
        "bot-left": hsv[h//2:, :w//2],
        "bot-right": hsv[h//2:, w//2:],
    }
    for name, region in regions.items():
        H = region[:, :, 0]
        S = region[:, :, 1]
        V = region[:, :, 2]
        print(f"  {name}: H=[{H.min()}-{H.max()}], S=[{S.min()}-{S.max()}], V=[{V.min()}-{V.max()}]")

    # Test ball detection with current thresholds
    from env.preprocessing import BallTracker
    tracker = BallTracker()
    bx, by = tracker.detect(img_rgb)
    print(f"  Ball detected at: ({bx:.3f}, {by:.3f})")

    # Save visualization
    vis = img_bgr.copy()
    if bx >= 0:
        cx, cy = int(bx * w), int(by * h)
        cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(vis, f"ball ({bx:.2f},{by:.2f})", (cx+10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Show HSV mask
    hsv_frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_frame, tracker.lower_yellow, tracker.upper_yellow)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    combined = np.vstack([vis, mask_bgr * 255])
    out_path = os.path.join(OUTPUT_DIR, f"analysis_{fname}")
    cv2.imwrite(out_path, combined)
    print(f"  Saved analysis to {out_path}")

print("\nDone. Open the analysis images in", OUTPUT_DIR)
