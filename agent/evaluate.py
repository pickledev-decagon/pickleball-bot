"""
Legacy evaluate.py — superseded by agent.competition_export.py.

This file re-exports the new AsyncPolicy-based evaluation for backward
compatibility. Import here will use the correct preprocessing (frame
stacking, grayscale, async inference, left/right support).

For new code, prefer:
    from agent.competition_export import TeamRight, TeamLeft
"""

from agent.competition_export import TeamRight, TeamLeft, AsyncPolicy


def create_agent_from_checkpoint(model_path, device="cpu", play_as_left=False,
                                  frame_stack=16):
    """
    Create a policy function from a trained checkpoint.

    DEPRECATED: Use TeamRight/TeamLeft directly instead.

    Args:
        model_path: Path to SB3 PPO checkpoint.
        device: "cpu" or "cuda".
        play_as_left: True for left-side agent (applies flip + mirror).
        frame_stack: Must match training config.

    Returns:
        Callable policy(observation, reward) -> np.array([v,h,r])
    """
    agent = AsyncPolicy(
        model_path=model_path,
        device=device,
        play_as_left=play_as_left,
        frame_stack=frame_stack,
    )
    return agent.policy
