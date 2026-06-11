import os
import sys
import time
import json
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from agent.wrapper_vec import create_vec_env
from agent.buffer import OpponentBuffer
from agent.reward_shaping import shaped_reward, HitDetector
from agent.selfplay_callback import SelfPlayCheckpointCallback
from config import (
    BUILD_PATH, NUM_ENVS, FRAME_STACK,
    STAGE1_FRAC, STAGE2_FRAC, STAGE3_FRAC,
    TIMESTEPS_PER_STAGE, TOTAL_TIMESTEPS,
    CHECKPOINT_DIR as SAVE_DIR, LOG_DIR, BUFFER_DIR, BUFFER_MAX_SIZE,
)

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(BUFFER_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

buffer = OpponentBuffer(BUFFER_DIR, max_size=5)


def square_motion_policy(obs, step_counter):
    side = (step_counter[0] // 30) % 4
    dirs = [
        np.array([1, 0, 0], dtype=np.int32),
        np.array([0, 1, 1], dtype=np.int32),
        np.array([2, 0, 2], dtype=np.int32),
        np.array([0, 2, 0], dtype=np.int32),
    ]
    step_counter[0] += 1
    return dirs[side]


class CurriculumCallback(BaseCallback):
    """
    SB3 callback that implements 3-stage curriculum:
      Stage 1 (0-30%): no-op opponent + dense reward
      Stage 2 (30-60%): square-motion opponent + fading dense reward
      Stage 3 (60-100%): self-play from opponent buffer + sparse only
    """
    def __init__(self, total_steps, buffer, buffer_dir, verbose=0):
        super().__init__(verbose)
        self.total_steps = total_steps
        self.buffer = buffer
        self.buffer_dir = buffer_dir
        self.stage = 0
        self.square_step_count = [0]
        self.hit_detector = HitDetector()
        self.opp_model = None
        self.opp_ckpt = None
        self._current_opp_fn = None

    def get_stage(self, progress):
        if progress < STAGE1_FRAC:
            return 1
        elif progress < STAGE1_FRAC + STAGE2_FRAC:
            return 2
        else:
            return 3

    def _get_wrapper(self, env):
        from env.wrapper import EnhancedUnityWrapper
        while hasattr(env, 'env') and not isinstance(env, EnhancedUnityWrapper):
            env = env.env
        return env

    def _on_training_start(self):
        # Initialize wrappers at the beginning of training
        self._update_envs(force=True)

    def _on_step(self):
        progress = self.num_timesteps / self.total_steps if self.total_steps > 0 else 0
        new_stage = self.get_stage(progress)

        if new_stage != self.stage:
            self.stage = new_stage
            if self.verbose > 0:
                print(f"\n=== Stage {self.stage} at step {self.num_timesteps} ===")
            self._update_envs(force=True)
        else:
            # Keep progress updated on every step even within same stage
            self._update_envs(force=False)

        return True

    def _update_envs(self, force=False):
        progress = self.num_timesteps / self.total_steps if self.total_steps > 0 else 0
        opp_fn = self._get_opponent_fn(self.stage, force)

        for env in self.model.env.envs:
            wrapper = self._get_wrapper(env)
            wrapper.opponent_policy = opp_fn
            wrapper.training_progress = progress

    def _get_opponent_fn(self, stage, force_refresh):
        if not force_refresh and self._current_opp_fn is not None:
            return self._current_opp_fn

        if stage == 1:
            fn = lambda obs: np.array([0, 0, 0], dtype=np.int32)
        elif stage == 2:
            counter = self.square_step_count
            def _square_policy(obs):
                return square_motion_policy(obs, counter)
            fn = _square_policy
        else:
            ckpt = self.buffer.sample()
            if ckpt is None:
                # Buffer empty — fall back to no-op until checkpoints are added
                fn = lambda obs: np.array([0, 0, 0], dtype=np.int32)
            else:
                path = os.path.join(self.buffer_dir, ckpt)
                if self.opp_ckpt != ckpt or self.opp_model is None:
                    self.opp_model = PPO.load(path, device="cpu")
                    self.opp_ckpt = ckpt
                model = self.opp_model
                def _buffer_policy(obs):
                    obs_batch = np.expand_dims(obs, axis=0)
                    action, _ = model.predict(obs_batch, deterministic=True)
                    return action[0]
                fn = _buffer_policy

        self._current_opp_fn = fn
        return fn


def train():
    print(f"Creating {NUM_ENVS} parallel Unity environments...")
    vec_env = create_vec_env(
        build_path=BUILD_PATH,
        num_envs=NUM_ENVS,
        frame_stack=FRAME_STACK,
        use_ball_tracking=True,
    )

    curriculum_callback = CurriculumCallback(
        total_steps=TOTAL_TIMESTEPS,
        buffer=buffer,
        buffer_dir=BUFFER_DIR,
    )

    model = PPO(
        "CnnPolicy",
        vec_env,
        learning_rate=2.5e-4,
        n_steps=2048 // NUM_ENVS,
        batch_size=128,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cpu",
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    checkpoint_callback = SelfPlayCheckpointCallback(
        save_freq=max(TOTAL_TIMESTEPS // 10, 5000),
        save_path=SAVE_DIR,
        buffer=buffer,
        buffer_dir=BUFFER_DIR,
        name_prefix=f"curriculum_{timestamp}",
        total_timesteps=TOTAL_TIMESTEPS,
    )

    print(f"Training: {TOTAL_TIMESTEPS} total steps, "
          f"{NUM_ENVS} parallel envs")
    print(f"  Stage 1 (0-{int(STAGE1_FRAC*100)}%): no-op opponent + dense reward")
    print(f"  Stage 2 ({int(STAGE1_FRAC*100)}-{int((STAGE1_FRAC+STAGE2_FRAC)*100)}%): square-motion opponent")
    print(f"  Stage 3 ({int((STAGE1_FRAC+STAGE2_FRAC)*100)}-100%): self-play opponent buffer")

    start = time.time()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[curriculum_callback, checkpoint_callback],
        tb_log_name=f"curriculum_{timestamp}",
        reset_num_timesteps=True,
    )
    elapsed = time.time() - start

    final_name = f"curriculum_final_{timestamp}.zip"
    final_path = os.path.join(BUFFER_DIR, final_name)
    model.save(final_path)

    buffer.add(final_name, win_rate=0.5)

    vec_env.close()

    print(f"\nTraining complete.")
    print(f"  Total steps: {TOTAL_TIMESTEPS} across {NUM_ENVS} envs")
    print(f"  Wall time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Final model: {final_path}")
    print(f"  Buffer checkpoints: {buffer.list_checkpoints()}")

    return final_path


if __name__ == "__main__":
    train()
