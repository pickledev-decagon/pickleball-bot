import numpy as np
import cv2
import os
import sys
import time
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
from mlagents_envs.envs.custom_side_channel import CustomDataChannel, StringSideChannel

BUILD_PATH = "/Users/keven/Downloads/dPickleball BuildFiles/Training/Mac/Mac.app"
SAVE_DIR = os.path.join(os.path.dirname(__file__), "captured_frames")
os.makedirs(SAVE_DIR, exist_ok=True)

string_channel = StringSideChannel()
channel = CustomDataChannel()
channel.send_data(serve=212, p1=0, p2=0)

unity_env = UnityEnvironment(BUILD_PATH, side_channels=[string_channel, channel])
env = UnityParallelEnv(unity_env)
env.reset()

square_directions = [
    np.array([1, 0, 0], dtype=np.int32),
    np.array([0, 1, 1], dtype=np.int32),
    np.array([2, 0, 2], dtype=np.int32),
    np.array([0, 2, 0], dtype=np.int32)
]
steps_per_side = 30

frame_count = 0
max_frames = 200

try:
    while env.agents and frame_count < max_frames:
        side = (frame_count // steps_per_side) % 4
        action = square_directions[side]
        actions = {env.agents[0]: action, env.agents[1]: action}
        observation, reward, done, info = env.step(actions)

        obs = observation[env.agents[0]]['observation'][0]
        img = np.transpose(obs, (1, 2, 0))

        if frame_count % 10 == 0:
            path = os.path.join(SAVE_DIR, f"frame_{frame_count:04d}.png")
            cv2.imwrite(path, cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            print(f"Saved {path}")

        if done[env.agents[0]] or done[env.agents[1]]:
            break

        frame_count += 1

finally:
    env.close()
    print(f"Captured {frame_count} frames to {SAVE_DIR}/")
