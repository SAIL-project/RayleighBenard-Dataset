import gymnasium as gym
import torch
print(torch.cuda.is_available())
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

env = gym.make("CartPole-v1", render_mode="rgb_array")

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# vec_env = make_vec_env("CartPole-v1", n_envs=4)
# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)