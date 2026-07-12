import numpy as np
from collections import deque


def annealing_weight(progress, start_phase=0.0, end_phase=0.6):
    if progress < start_phase:
        return 1.0
    if progress > end_phase:
        return 0.0
    return 1.0 - (progress - start_phase) / (end_phase - start_phase)


def dense_distance_reward(ball_pos, paddle_x=0.65, paddle_y=None, paddle_pos_y=None):
    """Dense proximity reward: penalize distance between ball and paddle.

    Coefficient 0.05 (was 0.01) — 5x increase to make proximity signal
    comparable to hit_bonus over an approach sequence.

    Magnitude analysis:
      - hit_bonus = 1.0 (one-time, on contact)
      - proximity per step at dist=0.5: -0.5 * 0.05 = -0.025
      - Over 50-step approach (avg dist ~0.25): sum ≈ -0.025 * 50 = -1.25
      - This makes "approach the ball" worth ~1.25, comparable to hit_bonus=1.0
      - Without this, agent learns to HIT (+1.0) but not to MOVE (+0.01/step)
      - 0.05 chosen over 0.1 to avoid reward hacking (chasing ball without hitting)
    """
    if ball_pos[0] < 0:
        return 0.0
    if paddle_y is None:
        paddle_y = ball_pos[1]
    effective_y = paddle_pos_y if paddle_pos_y is not None else paddle_y
    dx = ball_pos[0] - paddle_x
    dy = ball_pos[1] - effective_y
    dist = np.sqrt(dx**2 + dy**2)
    return -dist * 0.05


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
    # Compute full 2D velocity change magnitude to catch both
    # horizontal drives and vertical dinks.
    dv1 = np.array([recent[1][0] - recent[0][0], recent[1][1] - recent[0][1]])
    dv2 = np.array([recent[2][0] - recent[1][0], recent[2][1] - recent[1][1]])
    velocity_change = np.linalg.norm(dv2 - dv1)
    return velocity_change > threshold


class HitDetector:
    def __init__(self, window=5):
        self.trajectory = deque(maxlen=window)
        self.last_hit_step = -100
        self._ball_on_opp_recently = False  # track if ball came from opponent
        self.last_hit_was_return = False     # was the last hit a return (incoming ball)?
        # Effective hit tracking (Fix 1): ball crossed net after our hit
        self.prev_ball_y = -1.0
        self.effective_hit = False   # True on the step ball crosses to opponent's side
        self.out_of_bounds = False   # True on the step ball goes off court sides
        self._oob_penalized = False  # only penalize OOB once per hit
        # Rally counter (Fix 5): increments on each return, resets on point end
        self.rally_count = 0

    def reset(self):
        self.trajectory.clear()
        self.last_hit_step = -100
        self._ball_on_opp_recently = False
        self.last_hit_was_return = False
        self.prev_ball_y = -1.0
        self.effective_hit = False
        self.out_of_bounds = False
        self._oob_penalized = False
        self.rally_count = 0

    def update(self, ball_pos, step):
        # Reset per-step flags
        self.effective_hit = False
        self.out_of_bounds = False

        if ball_pos[0] >= 0:
            curr_y = ball_pos[1]

            # Check if ball crossed net (y=0.5) upward after our recent hit
            if (self.prev_ball_y >= 0 and
                self.prev_ball_y < 0.5 and
                curr_y >= 0.5 and
                (step - self.last_hit_step) <= 15):  # within 15 steps of our hit
                self.effective_hit = True
                if self.last_hit_was_return:
                    self.rally_count += 1

            # Check if ball went out of bounds (x off court) after our recent hit
            # Only fire once per hit (use _oob_penalized flag)
            if (not self._oob_penalized and
                (ball_pos[0] <= 0.02 or ball_pos[0] >= 0.98) and
                (step - self.last_hit_step) <= 15):
                self.out_of_bounds = True
                self._oob_penalized = True

            # Track ball side for return detection
            if curr_y > 0.5:
                self._ball_on_opp_recently = True
            elif curr_y < 0.3:
                pass  # _ball_on_opp_recently stays True until a hit resets it

            self.prev_ball_y = curr_y

        self.trajectory.append(ball_pos)
        if detect_paddle_hit(self.trajectory) and (step - self.last_hit_step) > 10:
            self.last_hit_step = step
            self.last_hit_was_return = self._ball_on_opp_recently
            self._ball_on_opp_recently = False  # reset after hit
            self._oob_penalized = False  # reset OOB flag for new hit
            return True
        return False


def shaped_reward(sparse_reward, ball_pos, training_progress,
                  step=0, hit_detector=None,
                  paddle_x=0.65, paddle_y=None,
                  aggression_bonus=0.0, hit_bonus=0.5,
                  proximity_weight=1.0, placement_bonus=0.0,
                  paddle_pos_y=None, rotation_penalty=0.0,
                  rotation_action=1,
                  return_bonus=0.0,
                  # NEW params (backward compatible — old configs unaffected)
                  effective_hit_only=True,  # if True, hit_bonus only when ball crosses net
                  contact_bonus=0.1,        # small reward for any contact (effective_hit_only mode)
                  oob_penalty=0.5,          # penalty for hitting ball out of bounds
                  rally_bonus=0.1):         # extra return bonus per rally shot
    """Shaped reward for dPickleBall training.

    When effective_hit_only=True (FIXED reward, Step A):
      - contact_bonus (0.1) for any paddle-ball contact
      - hit_bonus (1.0) ONLY when ball crosses to opponent's court (effective hit)
      - oob_penalty (-0.5) when ball goes out of bounds after hit
      - return_bonus scales with rally length: return_bonus + rally_bonus * rally_count

    When effective_hit_only=False (OLD reward, Step B baseline):
      - hit_bonus (1.0) for ANY contact (original behavior)
      - No contact_bonus, no effective hit check, no OOB penalty
    """
    dense_weight = annealing_weight(training_progress, end_phase=0.6)
    reward = sparse_reward

    # Reset rally counter on point end (Fix 5)
    if sparse_reward != 0 and hit_detector is not None:
        hit_detector.rally_count = 0

    if dense_weight > 0:
        reward += dense_weight * proximity_weight * dense_distance_reward(
            ball_pos, paddle_x, paddle_y, paddle_pos_y
        )
        if hit_detector is not None:
            hit_detected = hit_detector.update(ball_pos, step)
            if hit_detected:
                if effective_hit_only:
                    # Fixed reward: small contact bonus, main bonus on effective hit
                    reward += dense_weight * contact_bonus
                    # Return bonus scales with rally length
                    if return_bonus > 0 and hit_detector.last_hit_was_return:
                        reward += dense_weight * (return_bonus + rally_bonus * hit_detector.rally_count)
                else:
                    # Old reward: full hit_bonus for any contact
                    reward += dense_weight * hit_bonus
                    if return_bonus > 0 and hit_detector.last_hit_was_return:
                        reward += dense_weight * return_bonus

            # Effective hit bonus (ball crossed to opponent's court)
            if effective_hit_only and hit_detector.effective_hit:
                reward += dense_weight * hit_bonus

            # Out of bounds penalty
            if effective_hit_only and hit_detector.out_of_bounds:
                reward -= dense_weight * oob_penalty

        if placement_bonus > 0:
            # FIX: Use hit_detector's last_hit_step instead of sparse_reward.
            # The old code `last_hitter_was_me = (sparse_reward > 0)` was wrong
            # because sparse_reward is only nonzero when a POINT is scored, not
            # on every hit. During rallies (no point scored), sparse_reward=0,
            # so placement was never rewarded — defeating its purpose.
            # New logic: we were the last hitter if a hit was detected recently
            # (within the effective-hit window of 15 steps).
            if hit_detector is not None:
                last_hitter_was_me = (step - hit_detector.last_hit_step) <= 15
            else:
                last_hitter_was_me = (sparse_reward > 0)  # fallback if no detector
            reward += dense_weight * placement_bonus * placement_reward(ball_pos, last_hitter_was_me)
    if aggression_bonus > 0 and ball_pos[1] > 0.5:
        reward += aggression_bonus

    if rotation_penalty > 0 and rotation_action != 0:
        reward -= rotation_penalty

    return reward
