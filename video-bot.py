import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import os

video_folder = "videos"
os.makedirs(video_folder, exist_ok=True)

env = DummyVecEnv([lambda: gym.make("Acrobot-v1", render_mode="rgb_array")])
env = VecVideoRecorder(env, video_folder, record_video_trigger=lambda x: True, video_length=500, name_prefix="acrobot")

model = PPO.load("models/ppo_acrobot")

obs = env.reset()
for _ in range(500):
    action, _ = model.predict(obs)
    obs, _, done, _ = env.step(action)
    if done:
        break

env.close()
