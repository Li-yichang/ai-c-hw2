import gymnasium as gym
from stable_baselines3 import PPO
import os

env = gym.make("Acrobot-v1")
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_acrobot_tensorboard/")
model.learn(total_timesteps=100_000)

os.makedirs("models", exist_ok=True)
model.save("models/ppo_acrobot")
