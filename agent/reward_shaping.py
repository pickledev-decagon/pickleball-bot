import numpy as np
from collections import deque


def annealing_weight(progress, start_phase=0.0, end_phase=0.6):
    if progress < start_phase:
        return 1.0
    if progress > end_phase:
        return 0.0
    return 1.0 - (progress - start_phase) / (end_phase - start_phase)


def dense_distance_reward(ball_pos, paddle_x=0.65, paddle_y=None):
    if ball_pos[0] < 0:
        return 0.0
    # If actual paddle y-position is unknown, use ball's y as an estimate.
    # The paddle naturally tracks the ball vertically, so dy ≅ 0.
    # This avoids providing incorrect vertical signal (the old hardcoded
    # paddle_y=0.5 penalized the ball for being high or low on the court).
    if paddle_y is None:
        paddle_y = ball_pos[1]
    dx = ball_pos[0] - paddle_x
    dy = ball_pos[1] - paddle_y
    dist = np.sqrt(dx**2 + dy**2)
    return -dist * 0.01


def get_ball_land_zone(ball_x, ball_y, court_width=168, court_height=84):
    """
    Categorize where the ball lands on the opponent's side.
    Coordinates are normalized [0,1] where y=0.5 is the net.
    Returns zone name used for placement reward scoring.
    """
    if ball_y < 0.5:
        return "my_side"
    x_zone = "left" if ball_x < 0.3 else "right" if ball_x > 0.7 else "center"
    y_zone = "near_net" if ball_y < 0.7 else "far_back"
    if x_zone in ("left", "right") and y_zone == "far_back":
        return "corner"
    return f"{x_zone}_{y_zone}"


def placement_reward(ball_pos, last_hitter_was_me=True):
    """
    Reward for ball placement quality on opponent's side.
    Corner shots are hardest to return; near-net shots are easiest.
    """
    if not last_hitter_was_me or ball_pos[0] < 0:
        return 0.0
    zone = get_ball_land_zone(ball_pos[0], ball_pos[1])
    scores = {
        "corner": 0.5,
        "left_far_back": 0.3,
        "right_far_back": 0.3,
        "center_far_back": 0.2,
        "near_net": 0.1,
    }
    return scores.get(zone, 0.0)


def detect_paddle_hit(ball_trajectory, threshold=0.02):
    if len(ball_trajectory) < 3:
        return False
    recent = list(ball_trajectory)[-3:]
    if any(p[0] < 0 for p in recent):
        return False
    dx1 = recent[1][0] - recent[0][0]
    dx2 = recent[2][0] - recent[1][0]
    velocity_change = abs(dx2 - dx1)
    return velocity_change > threshold


class HitDetector:
    def __init__(self, window=5):
        self.trajectory = deque(maxlen=window)
        self.last_hit_step = -100

    def update(self, ball_pos, step):
        self.trajectory.append(ball_pos)
        if detect_paddle_hit(self.trajectory) and (step - self.last_hit_step) > 10:
            self.last_hit_step = step
            return True
        return False


def shaped_reward(sparse_reward, ball_pos, training_progress,
                  step=0, hit_detector=None,
                  paddle_x=0.65, paddle_y=None,
                  aggression_bonus=0.0, hit_bonus=0.5,
                  proximity_weight=1.0, placement_bonus=0.0):
    dense_weight = annealing_weight(training_progress, end_phase=0.6)
    reward = sparse_reward
    if dense_weight > 0:
        reward += dense_weight * proximity_weight * dense_distance_reward(ball_pos, paddle_x, paddle_y)
        if hit_detector is not None and hit_detector.update(ball_pos, step):
            reward += dense_weight * hit_bonus
        if placement_bonus > 0:
            # Simple heuristic: if ball is on opponent side and we got positive reward,
            # we likely just hit it. Reward placement quality.
            last_hitter_was_me = (sparse_reward > 0 or ball_pos[0] > 0.5)
            reward += dense_weight * placement_bonus * placement_reward(ball_pos, last_hitter_was_me)
    if aggression_bonus > 0 and ball_pos[0] > 0.5:
        reward += aggression_bonus
    return reward
