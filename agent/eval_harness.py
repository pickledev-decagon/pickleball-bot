"""
Win-rate evaluation harness for dPickleBall.

Plays full matches (to 21 points) against various opponent types and reports
win rates. This is the ground-truth metric for competition readiness.

Usage:
    from agent.eval_harness import MatchEvaluator
    evaluator = MatchEvaluator(model, lambda: make_unity_env_fn(...))
    result = evaluator.evaluate_vs("no_op", n_matches=100)
    print(result["win_rate"])
"""

import numpy as np
from collections import defaultdict


class MatchEvaluator:
    """
    Evaluates a trained PPO model by playing full matches to 21 points.
    """

    def __init__(self, model, env_fn, max_steps_per_match=10_000):
        """
        Args:
            model: Trained SB3 PPO model with .predict() method.
            env_fn: Callable that returns a single EnhancedUnityWrapper instance.
            max_steps_per_match: Hard cap to prevent infinite rallies.
        """
        self.model = model
        self.env_fn = env_fn
        self.max_steps_per_match = max_steps_per_match

    def evaluate_vs(self, opponent_type="no_op", n_matches=50, deterministic=True,
                    env_fn=None):
        """
        Play n_matches against a specific opponent type.

        Args:
            opponent_type: "no_op", "random", "square", or "self".
            n_matches: Number of matches to play.
            deterministic: Whether model.predict uses deterministic=True.
            env_fn: Callable that returns an EnhancedUnityWrapper.
                    Defaults to self.env_fn. Pass a left-side env_fn
                    (make_unity_env_fn(..., play_as_left=True)) to test
                    left-side play.

        Returns dict with:
            win_rate, wins, losses, draws, avg_match_len, avg_point_diff, side
        """
        env_fn = env_fn or self.env_fn
        results = defaultdict(int)
        total_steps = 0
        total_point_diff = 0.0
        # Detect which side this env plays (right by default)
        side = "unknown"
        try:
            test_env = env_fn()
            side = "left" if getattr(test_env, 'play_as_left', False) else "right"
            test_env.close()
        except Exception:
            pass

        # Create ONE environment and reuse it across matches via reset()
        env = env_fn()
        try:
            for match_i in range(n_matches):
                # Inject opponent policy
                if opponent_type == "no_op":
                    env.opponent_policy = lambda obs: np.array([1, 1, 1], dtype=np.int32)
                elif opponent_type == "random":
                    env.opponent_policy = lambda obs: env.action_space.sample()
                elif opponent_type == "square":
                    seq_step = [0]
                    env.opponent_policy = lambda obs: self._square_motion(seq_step)
                elif opponent_type == "self":
                    env.opponent_policy = lambda obs: self._self_play_policy(obs, deterministic)
                else:
                    raise ValueError(f"Unknown opponent_type: {opponent_type}")

                obs, info = env.reset()
                my_score = 0
                opp_score = 0
                match_steps = 0

                while match_steps < self.max_steps_per_match:
                    action, _ = self.model.predict(obs, deterministic=deterministic)
                    # Ensure valid discrete action
                    action = np.clip(action.astype(np.int32), 0, 2)

                    obs, reward, done, truncated, info = env.step(action)
                    match_steps += 1

                    # Extract score from info dict if available, else infer from reward
                    my_score = info.get("my_score", my_score)
                    opp_score = info.get("opp_score", opp_score)

                    # Fallback: infer from sparse reward spikes (point won/lost)
                    if reward > 0.8:
                        my_score += 1
                    elif reward < -0.8:
                        opp_score += 1

                    if done or truncated:
                        break

                    # Early termination if someone reaches 21
                    if my_score >= 21 or opp_score >= 21:
                        break

                if my_score > opp_score:
                    results["wins"] += 1
                elif my_score < opp_score:
                    results["losses"] += 1
                else:
                    results["draws"] += 1

                total_steps += match_steps
                total_point_diff += (my_score - opp_score)
        finally:
            env.close()

        n_matches = float(n_matches)
        win_rate = results["wins"] / n_matches if n_matches > 0 else 0.0
        avg_match_len = total_steps / n_matches if n_matches > 0 else 0.0
        avg_point_diff = total_point_diff / n_matches if n_matches > 0 else 0.0

        return {
            "opponent_type": opponent_type,
            "side": side,
            "n_matches": int(n_matches),
            "win_rate": round(win_rate, 4),
            "wins": results["wins"],
            "losses": results["losses"],
            "draws": results["draws"],
            "avg_match_len": round(avg_match_len, 1),
            "avg_point_diff": round(avg_point_diff, 2),
        }

    def benchmark_all(self, n_matches=50, deterministic=True, build_path=None):
        """
        Run evaluate_vs for all standard opponent types on BOTH sides.

        Requires build_path to create left-side envs. If build_path is None,
        uses the build path embedded in self.env_fn.

        Returns dict with:
            {"right": {opponent: results}, "left": {opponent: results}}
        """
        right_fn = self.env_fn

        # Create a left-side env factory
        from agent.wrapper_vec import make_unity_env_fn
        left_fn = make_unity_env_fn(
            rank=99, build_path=build_path or "/dev/null",
            play_as_left=True, frame_stack=16,
        )

        summary = {"right": {}, "left": {}}
        for side_name, env_fn in [("right", right_fn), ("left", left_fn)]:
            print(f"\n  Evaluating {side_name} side ...")
            for opp in ["no_op", "random", "square", "self"]:
                print(f"    vs {opp} ...")
                summary[side_name][opp] = self.evaluate_vs(
                    opp, n_matches, deterministic, env_fn=env_fn
                )
        return summary

    @staticmethod
    def _square_motion(step_counter):
        """Curriculum stage-2 square motion opponent."""
        side = (step_counter[0] // 30) % 4
        dirs = [
            np.array([1, 0, 0], dtype=np.int32),
            np.array([0, 1, 1], dtype=np.int32),
            np.array([2, 0, 2], dtype=np.int32),
            np.array([0, 2, 0], dtype=np.int32),
        ]
        step_counter[0] += 1
        return dirs[side]

    def _self_play_policy(self, obs, deterministic):
        """Use the same model for opponent inference."""
        # obs may need batch dimension
        if len(obs.shape) == 3:
            obs = np.expand_dims(obs, axis=0)
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return np.clip(action.astype(np.int32).flatten()[:3], 0, 2)
