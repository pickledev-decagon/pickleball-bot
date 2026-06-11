"""
Custom SB3 callback that saves checkpoints AND adds them to the opponent buffer
mid-training, so Stage 3 self-play has actual opponents to sample from.

Usage:
    from agent.selfplay_callback import SelfPlayCheckpointCallback
    callback = SelfPlayCheckpointCallback(
        save_freq=20_000,
        save_path="./checkpoints_selfplay_v2",
        buffer=opponent_buffer,
        buffer_dir="./opponent_buffer_v2",
        name_prefix="curriculum",
    )
"""

import os
import shutil
from stable_baselines3.common.callbacks import BaseCallback


class SelfPlayCheckpointCallback(BaseCallback):
    """
    Extends standard checkpoint saving with automatic opponent-buffer population.

    Every `save_freq` steps:
      1. Saves model to `save_path` (standard checkpoint)
      2. Copies checkpoint to `buffer_dir`
      3. Adds to OpponentBuffer with heuristic win_rate

    The heuristic win_rate is the training progress (0→1), which is a coarse
    proxy — later checkpoints are generally stronger. For true win-rate
    sorting, run `agent/eval_harness.py` offline and update the manifest.
    """

    def __init__(self, save_freq, save_path, buffer, buffer_dir,
                 name_prefix="model", total_timesteps=None, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.buffer = buffer
        self.buffer_dir = buffer_dir
        self.name_prefix = name_prefix
        self.total_timesteps = total_timesteps
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(buffer_dir, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            # 1. Standard checkpoint save
            ckpt_name = f"{self.name_prefix}_{self.num_timesteps}_steps.zip"
            ckpt_path = os.path.join(self.save_path, ckpt_name)
            self.model.save(ckpt_path)

            # 2. Copy to opponent buffer directory
            buffer_name = f"{self.name_prefix}_mid_{self.num_timesteps}.zip"
            buffer_path = os.path.join(self.buffer_dir, buffer_name)
            shutil.copy(ckpt_path, buffer_path)

            # 3. Heuristic win_rate = training progress
            #    More recent checkpoints are assumed stronger.
            #    Override offline with real win-rate eval for better sorting.
            max_steps = self.total_timesteps or max(self.num_timesteps, 1)
            heuristic_wr = min(self.num_timesteps / max_steps, 1.0)

            self.buffer.add(buffer_name, win_rate=heuristic_wr)

            if self.verbose > 0:
                print(f"\n[SelfPlayCheckpoint] Saved {ckpt_name}")
                print(f"[SelfPlayCheckpoint] Added to buffer: {buffer_name} "
                      f"(heuristic win_rate={heuristic_wr:.2f})")

        return True
