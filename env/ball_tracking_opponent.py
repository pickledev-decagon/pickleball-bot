"""
Ball-tracking scripted opponent for dPickleBall training.

Moves paddle toward the ball when it's on the opponent's side, returns to
center otherwise. Uses dead reckoning to estimate its own paddle position
(Unity doesn't expose opponent paddle coords). Stochastic to prevent overfitting.

Action order: [vertical, horizontal, rotation], 0=none, v=1=up, v=2=down,
h=1=right, h=2=left, r=1=ccw, r=2=cw.

Usage:
    from env.ball_tracking_opponent import BallTrackingOpponent
    opp = BallTrackingOpponent(hit_rate=0.6, noise_rate=0.2)
    # In wrapper: opp_action = opp(obs, ball_pos=(bx, by))
"""

import numpy as np


class BallTrackingOpponent:
    """Scripted opponent that tracks the ball with stochastic behavior."""

    def __init__(self,
                 hit_rate=0.6,
                 noise_rate=0.2,
                 reaction_delay=3,
                 paddle_step=0.008,
                 start_x=0.35,
                 start_y=0.5,
                 seed=None):
        """
        Args:
            hit_rate: Probability of actually moving toward ball (vs missing).
            noise_rate: Probability of random action instead of tracking.
            reaction_delay: Steps to wait before reacting to ball coming over.
            paddle_step: Estimated paddle movement per step (dead reckoning).
            start_x: Initial paddle x (0.35 if opponent is left side).
            start_y: Initial paddle y (0.5 = center).
            seed: Random seed for reproducibility.
        """
        self.hit_rate = hit_rate
        self.noise_rate = noise_rate
        self.reaction_delay = reaction_delay
        self.paddle_step = paddle_step
        self.rng = np.random.RandomState(seed)

        # Dead-reckoned paddle position
        self.paddle_x = start_x
        self.paddle_y = start_y

        # Reaction delay tracking
        self._ball_coming_steps = 0  # steps since ball entered opponent's side
        self._last_ball_side = "agent"  # "agent" or "opponent"

    def reset(self):
        """Reset paddle position estimate and delay counter."""
        self.paddle_x = 0.35
        self.paddle_y = 0.5
        self._ball_coming_steps = 0
        self._last_ball_side = "agent"

    def __call__(self, obs, ball_pos=None, paddle_pos=None):
        """Compute opponent action. ball_pos is (x, y) in normalized [0,1] coords.
        
        Args:
            obs: Stacked observation (unused by scripted opponents).
            ball_pos: (x, y) normalized ball position from visual detection.
            paddle_pos: (x, y) normalized opponent paddle position from visual
                       detection. If provided, replaces dead reckoning estimate.
        """
        # Update paddle position from visual detection if available
        if paddle_pos is not None and paddle_pos[0] >= 0:
            self.paddle_x, self.paddle_y = paddle_pos

        # Default: no action
        v, h, r = 0, 0, 0

        # If no ball position available, return no-op
        if ball_pos is None or ball_pos[0] < 0:
            self._integrate_position(0, 0)
            return np.array([0, 0, 0], dtype=np.int32)

        bx, by = ball_pos

        # Determine which side the ball is on
        # y > 0.5 = opponent's side (opponent needs to hit it back)
        ball_on_opp_side = by > 0.5

        if ball_on_opp_side:
            # Ball is coming toward opponent — track it
            if self._last_ball_side != "opponent":
                self._ball_coming_steps = 0
                self._last_ball_side = "opponent"
            self._ball_coming_steps += 1

            # Apply reaction delay
            if self._ball_coming_steps <= self.reaction_delay:
                self._integrate_position(0, 0)
                return np.array([0, 0, 0], dtype=np.int32)

            # Stochastic: sometimes miss the ball
            if self.rng.random() > self.hit_rate:
                self._integrate_position(0, 0)
                return np.array([0, 0, 0], dtype=np.int32)

            # Stochastic: sometimes make random move (noise)
            if self.rng.random() < self.noise_rate:
                v = self.rng.choice([0, 1, 2])
                h = self.rng.choice([0, 1, 2])
                r = self.rng.choice([0, 0, 0, 1, 2])  # mostly no rotation
                self._integrate_position(v, h)
                return np.array([v, h, r], dtype=np.int32)

            # Track ball: move paddle toward ball position
            # Vertical: move up if ball above paddle, down if below
            if by > self.paddle_y + 0.02:
                v = 1  # up
            elif by < self.paddle_y - 0.02:
                v = 2  # down

            # Horizontal: move toward ball x
            if bx > self.paddle_x + 0.02:
                h = 1  # right
            elif bx < self.paddle_x - 0.02:
                h = 2  # left

            # Occasionally add rotation for variety
            if self.rng.random() < 0.15:
                r = self.rng.choice([1, 2])

        else:
            # Ball is on agent's side — return to center vertically, keep x.
            # Don't move horizontally: paddle stays on its side of the court
            # (x is set from visual detection each step).
            self._last_ball_side = "agent"
            self._ball_coming_steps = 0

            if self.paddle_y > 0.52:
                v = 2  # down
            elif self.paddle_y < 0.48:
                v = 1  # up

        self._integrate_position(v, h)
        return np.array([v, h, r], dtype=np.int32)

    def _integrate_position(self, v, h):
        """Update dead-reckoned paddle position based on action."""
        if v == 1:  # up
            self.paddle_y = min(1.0, self.paddle_y + self.paddle_step)
        elif v == 2:  # down
            self.paddle_y = max(0.0, self.paddle_y - self.paddle_step)

        if h == 1:  # right
            self.paddle_x = min(1.0, self.paddle_x + self.paddle_step)
        elif h == 2:  # left
            self.paddle_x = max(0.0, self.paddle_x - self.paddle_step)


def make_ball_tracking_opponent(hit_rate=0.6, noise_rate=0.2, reaction_delay=3, seed=None):
    """Factory that returns a BallTrackingOpponent instance."""
    return BallTrackingOpponent(
        hit_rate=hit_rate,
        noise_rate=noise_rate,
        reaction_delay=reaction_delay,
        seed=seed,
    )
