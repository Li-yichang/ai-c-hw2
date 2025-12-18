import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
import os

# 建立錄影資料夾
video_folder = "videos"
os.makedirs(video_folder, exist_ok=True)

# 包裝環境以支援錄影
env = gym.make("CartPole-v1", render_mode="rgb_array")
env = DummyVecEnv([lambda: env])
env = VecVideoRecorder(env, video_folder, record_video_trigger=lambda x: True, video_length=500, name_prefix="cartpole")

# 載入 model
model = PPO.load("models/ppo_cartpole")

obs = env.reset()
for _ in range(500):
    action, _ = model.predict(obs)
    obs, _, dones, _ = env.step(action)
    if dones:
        break

env.close()
