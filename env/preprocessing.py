import numpy as np
import cv2


class BallTracker:
    def __init__(self):
        # HSV range for standard yellow pickleball
        # Tune these empirically based on actual env frames
        self.lower_yellow = np.array([20, 80, 80], dtype=np.uint8)
        self.upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
        self.min_ball_area = 5
        self.max_ball_area = 200
        self.prev_ball_pos = (-1.0, -1.0)

    def set_hsv_range(self, lower, upper):
        self.lower_yellow = np.array(lower, dtype=np.uint8)
        self.upper_yellow = np.array(upper, dtype=np.uint8)

    def detect(self, frame_rgb):
        if frame_rgb is None or frame_rgb.size == 0:
            return (-1.0, -1.0)

        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

        # Morphological ops to clean up noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.prev_ball_pos = (-1.0, -1.0)
            return (-1.0, -1.0)

        # Filter by area and find largest valid contour
        valid = [c for c in contours if self.min_ball_area < cv2.contourArea(c) < self.max_ball_area]
        if not valid:
            self.prev_ball_pos = (-1.0, -1.0)
            return (-1.0, -1.0)

        largest = max(valid, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            self.prev_ball_pos = (-1.0, -1.0)
            return (-1.0, -1.0)

        h, w = frame_rgb.shape[:2]
        cx = M["m10"] / M["m00"] / w
        cy = M["m01"] / M["m00"] / h
        self.prev_ball_pos = (cx, cy)
        return (cx, cy)


def frame_diff(prev_frame, curr_frame, threshold=30):
    if prev_frame is None or curr_frame is None:
        return None
    if len(prev_frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
    else:
        prev_gray = prev_frame
        curr_gray = curr_frame
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return thresh.astype(np.float32) / 255.0


def detect_paddles(frame_rgb, lower_color=None, upper_color=None):
    if lower_color is None:
        lower_color = np.array([100, 50, 50], dtype=np.uint8)  # blue paddles
    if upper_color is None:
        upper_color = np.array([130, 255, 255], dtype=np.uint8)
    # Use BGR2HSV — Unity frames are BGR (like OpenCV default), not RGB.
    # RGB2HSV was finding large static court elements instead of actual paddles.
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    positions = []
    for c in contours:
        area = cv2.contourArea(c)
        # Filter: exclude noise (too small) and static court elements (too large)
        # Actual paddles are ~50-200 area; static objects are ~900+
        if 20 < area < 500:
            x, y, w, h = cv2.boundingRect(c)
            cx = (x + w / 2) / frame_rgb.shape[1]
            cy = (y + h / 2) / frame_rgb.shape[0]
            positions.append((cx, cy, area))
    # Return 2 largest (left and right paddles)
    positions.sort(key=lambda p: p[2], reverse=True)
    results = [(-1.0, -1.0), (-1.0, -1.0)]
    for i, (cx, cy, _) in enumerate(positions[:2]):
        results[i] = (cx, cy)
    return results


class SmoothBallTracker:
    """
    Temporal smoothing wrapper around BallTracker.
    Uses simple velocity prediction to handle brief occlusions.
    """
    def __init__(self, alpha=0.7):
        self.last_pos = None
        self.velocity = np.array([0.0, 0.0])
        self.alpha = alpha  # velocity smoothing factor

    def update(self, detected_pos):
        """
        Args:
            detected_pos: (x, y) tuple from BallTracker, or (-1, -1) if miss.
        Returns:
            (x, y) tuple — predicted if occluded, detected otherwise.
        """
        if detected_pos[0] < 0:
            # Occlusion — predict from velocity
            if self.last_pos is not None:
                predicted = self.last_pos + self.velocity
                # Keep prediction in bounds
                predicted = np.clip(predicted, 0.0, 1.0)
                self.last_pos = predicted
                return (predicted[0], predicted[1])
            return (-1.0, -1.0)

        pos = np.array(detected_pos, dtype=np.float32)
        if self.last_pos is not None:
            raw_vel = pos - self.last_pos
            self.velocity = self.alpha * self.velocity + (1 - self.alpha) * raw_vel
        self.last_pos = pos
        return (pos[0], pos[1])
