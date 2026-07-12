"""
Specialist opponents for targeted weakness training.

KitchenServer: Hits balls that land near the net on agent's side (kitchen/
non-volley zone). Trains agent to move forward for short balls.

NetInterceptor: Rushes to net to intercept agent's serves, creating quick
returns. Trains agent to handle fast returns.

Both opponents can SERVE (unlike BallTrackingOpponent which only reacts).
Coordinate system: y > 0.5 = opponent's side, y < 0.5 = agent's side,
net at y = 0.5. Opponent paddle at x ≈ 0.35.

Action order: [vertical, horizontal, rotation], 0=none, v=1=up, v=2=down,
h=1=right, h=2=left, r=1=ccw, r=2=cw.

Usage:
    from env.specialist_opponents import KitchenServer, NetInterceptor
    opp = KitchenServer(hit_rate=0.9)
    # In wrapper: opp_action = opp(obs, ball_pos=(bx, by))
"""

import numpy as np


class KitchenServer:
    """Opponent that hits balls short — landing near the net on agent's side.

    Strategy: position paddle near the net (y ≈ 0.52-0.55) when hitting.
    Shorter hit distance + rotation spin = ball drops into the kitchen.
    This trains the agent to move forward and hit short balls.
    """

    def __init__(self,
                 hit_rate=0.9,
                 noise_rate=0.05,
                 reaction_delay=0,
                 paddle_step=0.008,
                 start_x=0.35,
                 start_y=0.5,
                 net_y=0.53,
                 seed=None):
        """
        Args:
            hit_rate: Probability of actually moving toward ball (vs missing).
            noise_rate: Probability of random action instead of tracking.
            reaction_delay: Steps to wait before reacting.
            paddle_step: Estimated paddle movement per step (dead reckoning).
            net_y: Target y position when hitting (just behind net on opp side).
            seed: Random seed.
        """
        self.hit_rate = hit_rate
        self.noise_rate = noise_rate
        self.reaction_delay = reaction_delay
        self.paddle_step = paddle_step
        self.net_y = net_y
        self.rng = np.random.RandomState(seed)

        self.paddle_x = start_x
        self.paddle_y = start_y
        self._ball_coming_steps = 0
        self._last_ball_side = "agent"
        self._hit_count = 0

    def reset(self):
        self.paddle_x = 0.35
        self.paddle_y = 0.5
        self._ball_coming_steps = 0
        self._last_ball_side = "agent"
        self._hit_count = 0

    def __call__(self, obs, ball_pos=None, paddle_pos=None):
        v, h, r = 0, 0, 0

        # Update paddle position from visual detection if available
        if paddle_pos is not None and paddle_pos[0] >= 0:
            self.paddle_x, self.paddle_y = paddle_pos

        if ball_pos is None or ball_pos[0] < 0:
            self._integrate_position(0, 0)
            return np.array([0, 0, 0], dtype=np.int32)

        bx, by = ball_pos
        ball_on_opp_side = by > 0.5

        if ball_on_opp_side:
            if self._last_ball_side != "opponent":
                self._ball_coming_steps = 0
                self._last_ball_side = "opponent"
            self._ball_coming_steps += 1

            if self._ball_coming_steps <= self.reaction_delay:
                self._integrate_position(0, 0)
                return np.array([0, 0, 0], dtype=np.int32)

            if self.rng.random() > self.hit_rate:
                self._integrate_position(0, 0)
                return np.array([0, 0, 0], dtype=np.int32)

            if self.rng.random() < self.noise_rate:
                v = self.rng.choice([0, 1, 2])
                h = self.rng.choice([0, 1, 2])
                r = self.rng.choice([0, 0, 0, 1, 2])
                self._integrate_position(v, h)
                return np.array([v, h, r], dtype=np.int32)

            # KEY DIFFERENCE: move toward net position (net_y) to hit short
            # Vertical: move to net_y position (just behind net on opp side)
            if self.paddle_y < self.net_y - 0.02:
                v = 1  # up (toward net/opponent's side boundary)
            elif self.paddle_y > self.net_y + 0.02:
                v = 2  # down

            # Horizontal: track ball x
            if bx > self.paddle_x + 0.02:
                h = 1  # right
            elif bx < self.paddle_x - 0.02:
                h = 2  # left

            # Use rotation to create spin for short drop
            # Alternate between cw and ccw for variety
            if self._hit_count % 2 == 0:
                r = 2  # cw
            else:
                r = 1  # ccw
            self._hit_count += 1

        else:
            # Ball on agent's side — return to center vertically, keep x.
            # Don't move horizontally: paddle stays on its side of the court.
            self._last_ball_side = "agent"
            self._ball_coming_steps = 0

            if self.paddle_y > 0.52:
                v = 2  # down
            elif self.paddle_y < 0.48:
                v = 1  # up

        self._integrate_position(v, h)
        return np.array([v, h, r], dtype=np.int32)

    def _integrate_position(self, v, h):
        if v == 1:
            self.paddle_y = min(1.0, self.paddle_y + self.paddle_step)
        elif v == 2:
            self.paddle_y = max(0.0, self.paddle_y - self.paddle_step)
        if h == 1:
            self.paddle_x = min(1.0, self.paddle_x + self.paddle_step)
        elif h == 2:
            self.paddle_x = max(0.0, self.paddle_x - self.paddle_step)


class NetInterceptor:
    """Opponent that rushes to the net to intercept agent's serves.

    Strategy: when ball is on agent's side (agent serving/hitting), rush to
    net position (y ≈ 0.52-0.55) and track ball x. When ball crosses to
    opponent's side, intercept immediately and hit back with power.
    This creates quick returns that train agent to react fast.
    """

    def __init__(self,
                 hit_rate=0.9,
                 noise_rate=0.05,
                 reaction_delay=0,
                 paddle_step=0.012,
                 start_x=0.35,
                 start_y=0.5,
                 net_y=0.53,
                 seed=None):
        """
        Args:
            hit_rate: Probability of actually moving toward ball.
            noise_rate: Probability of random action.
            reaction_delay: Steps to wait before reacting (0 = instant).
            paddle_step: Estimated paddle movement per step (HIGHER = faster rush).
            net_y: Target y position at net (just behind net on opp side).
            seed: Random seed.
        """
        self.hit_rate = hit_rate
        self.noise_rate = noise_rate
        self.reaction_delay = reaction_delay
        self.paddle_step = paddle_step  # Higher than KitchenServer — rush to net
        self.net_y = net_y
        self.rng = np.random.RandomState(seed)

        self.paddle_x = start_x
        self.paddle_y = start_y
        self._ball_coming_steps = 0
        self._last_ball_side = "agent"

    def reset(self):
        self.paddle_x = 0.35
        self.paddle_y = 0.5
        self._ball_coming_steps = 0
        self._last_ball_side = "agent"

    def __call__(self, obs, ball_pos=None, paddle_pos=None):
        v, h, r = 0, 0, 0

        # Update paddle position from visual detection if available
        if paddle_pos is not None and paddle_pos[0] >= 0:
            self.paddle_x, self.paddle_y = paddle_pos

        if ball_pos is None or ball_pos[0] < 0:
            self._integrate_position(0, 0)
            return np.array([0, 0, 0], dtype=np.int32)

        bx, by = ball_pos
        ball_on_opp_side = by > 0.5

        if ball_on_opp_side:
            # Ball crossed to opponent's side — INTERCEPT immediately
            if self._last_ball_side != "opponent":
                self._ball_coming_steps = 0
                self._last_ball_side = "opponent"
            self._ball_coming_steps += 1

            if self._ball_coming_steps <= self.reaction_delay:
                self._integrate_position(0, 0)
                return np.array([0, 0, 0], dtype=np.int32)

            if self.rng.random() > self.hit_rate:
                self._integrate_position(0, 0)
                return np.array([0, 0, 0], dtype=np.int32)

            if self.rng.random() < self.noise_rate:
                v = self.rng.choice([0, 1, 2])
                h = self.rng.choice([0, 1, 2])
                r = self.rng.choice([0, 0, 0, 1, 2])
                self._integrate_position(v, h)
                return np.array([v, h, r], dtype=np.int32)

            # Track ball position to intercept
            if by > self.paddle_y + 0.02:
                v = 1  # up
            elif by < self.paddle_y - 0.02:
                v = 2  # down

            if bx > self.paddle_x + 0.02:
                h = 1  # right
            elif bx < self.paddle_x - 0.02:
                h = 2  # left

            # Hit with rotation for power return
            r = self.rng.choice([1, 2])  # random spin for variety

        else:
            # KEY DIFFERENCE: Ball on agent's side — RUSH TO NET
            # (unlike BallTrackingOpponent which returns to center)
            self._last_ball_side = "agent"
            self._ball_coming_steps = 0

            # Rush to net position (net_y) to be ready for intercept
            if self.paddle_y < self.net_y - 0.02:
                v = 1  # up (toward net)
            elif self.paddle_y > self.net_y + 0.02:
                v = 2  # down

            # Track ball x to align for intercept
            if bx > self.paddle_x + 0.02:
                h = 1  # right
            elif bx < self.paddle_x - 0.02:
                h = 2  # left

        self._integrate_position(v, h)
        return np.array([v, h, r], dtype=np.int32)

    def _integrate_position(self, v, h):
        if v == 1:
            self.paddle_y = min(1.0, self.paddle_y + self.paddle_step)
        elif v == 2:
            self.paddle_y = max(0.0, self.paddle_y - self.paddle_step)
        if h == 1:
            self.paddle_x = min(1.0, self.paddle_x + self.paddle_step)
        elif h == 2:
            self.paddle_x = max(0.0, self.paddle_x - self.paddle_step)


def make_kitchen_server(hit_rate=0.9, noise_rate=0.05, reaction_delay=0, seed=None):
    """Factory that returns a KitchenServer instance."""
    return KitchenServer(
        hit_rate=hit_rate,
        noise_rate=noise_rate,
        reaction_delay=reaction_delay,
        seed=seed,
    )


def make_net_interceptor(hit_rate=0.9, noise_rate=0.05, reaction_delay=0, seed=None):
    """Factory that returns a NetInterceptor instance."""
    return NetInterceptor(
        hit_rate=hit_rate,
        noise_rate=noise_rate,
        reaction_delay=reaction_delay,
        seed=seed,
    )


class SpecialistOpponent:
    """Combined opponent that alternates between KitchenServer and NetInterceptor.

    Randomly picks one strategy on each reset. This ensures the agent sees
    both kitchen balls AND net intercepts during training without needing
    two separate training runs.
    """

    def __init__(self, hit_rate=0.9, noise_rate=0.05, reaction_delay=0, seed=None):
        self.rng = np.random.RandomState(seed)
        self.kitchen = KitchenServer(hit_rate=hit_rate, noise_rate=noise_rate,
                                      reaction_delay=reaction_delay, seed=seed)
        self.interceptor = NetInterceptor(hit_rate=hit_rate, noise_rate=noise_rate,
                                           reaction_delay=reaction_delay, seed=seed + 1 if seed else None)
        self._active = self.kitchen
        self._pick_active()

    def _pick_active(self):
        if self.rng.random() < 0.5:
            self._active = self.kitchen
        else:
            self._active = self.interceptor

    def reset(self):
        self._pick_active()
        self._active.reset()

    def __call__(self, obs, ball_pos=None, paddle_pos=None):
        return self._active(obs, ball_pos, paddle_pos)


def make_specialist_opponent(hit_rate=0.9, noise_rate=0.05, reaction_delay=0, seed=None):
    """Factory that returns a SpecialistOpponent (alternates kitchen/intercept)."""
    return SpecialistOpponent(
        hit_rate=hit_rate,
        noise_rate=noise_rate,
        reaction_delay=reaction_delay,
        seed=seed,
    )
