import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("CartPole-v1", render_mode="rgb_array")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

# 儲存 model
model.save("PPO/models/ppo_cartpole")

# 評估
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} ± {std_reward}")
