import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
from mlagents_envs.envs.custom_side_channel import CustomDataChannel, StringSideChannel
from stable_baselines3.common.vec_env import DummyVecEnv

from env.wrapper import EnhancedUnityWrapper


def make_unity_env_fn(rank, build_path, frame_stack=16, use_ball_tracking=True,
                      base_port=5005, serve_mode=212, vec_env_cls=DummyVecEnv,
                      use_rgb=False, use_frame_diff=False, use_heatmap=False,
                      reward_config=None, opponent_policy=None,
                      play_as_left=False,
                      heatmap_sigma=5, heatmap_weight=0.7): ##NEW
    def _thunk():
        string_channel = StringSideChannel()
        channel = CustomDataChannel()
        channel.send_data(serve=serve_mode, p1=0, p2=0)

        unity = UnityEnvironment(
            file_name=build_path,
            worker_id=rank,
            base_port=base_port,
            side_channels=[string_channel, channel],
            no_graphics=False,
        )
        parallel = UnityParallelEnv(unity)
        parallel.reset()

        env = EnhancedUnityWrapper(
            parallel,
            frame_stack=frame_stack,
            use_ball_tracking=use_ball_tracking,
            use_rgb=use_rgb,
            use_frame_diff=use_frame_diff,
            use_heatmap=use_heatmap,
            reward_config=reward_config,
            opponent_policy=opponent_policy,
            play_as_left=play_as_left,
            heatmap_sigma=heatmap_sigma, #NEW
            heatmap_weight=heatmap_weight, #NEW
        )
        return env
    return _thunk


def create_vec_env(build_path, num_envs=8, frame_stack=16,
                   use_ball_tracking=True, base_port=5005,
                   use_rgb=False, use_frame_diff=False, use_heatmap=False,
                   reward_config=None, opponent_policy=None,
                   heatmap_sigma=5, heatmap_weight=0.7): #NEW
    env_fns = [
        make_unity_env_fn(
            rank=i,
            build_path=build_path,
            frame_stack=frame_stack,
            use_ball_tracking=use_ball_tracking,
            base_port=base_port,
            use_rgb=use_rgb,
            use_frame_diff=use_frame_diff,
            use_heatmap=use_heatmap,
            reward_config=reward_config,
            opponent_policy=opponent_policy,
            play_as_left=(i >= num_envs // 2),  # half left, half right
            heatmap_sigma=heatmap_sigma, #NEW
            heatmap_weight=heatmap_weight, #NEW
        )
        for i in range(num_envs)
    ]
    return DummyVecEnv(env_fns)
