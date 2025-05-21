import gymnasium as gym
import torch
from stable_baselines3 import PPO
from multiprocessing import freeze_support

from main import create_env


class SwitchingPolicyEnv(gym.Wrapper):
    def __init__(self, env, model_inside, model_outside):
        super().__init__(env)
        self.model_inside = model_inside      # коли в центрі
        self.model_outside = model_outside    # коли поза центром
        self.last_obs = None

    def reset(self, **kwargs):
        self.last_obs, info = self.env.reset(**kwargs)
        return self.last_obs, info

    def step(self, action):
        self.last_obs, reward, done, truncated, info = self.env.step(action)
        return self.last_obs, reward, done, truncated, info

    def get_policy_action(self):
        obs_tensor = torch.tensor(self.last_obs, dtype=torch.float32).unsqueeze(0)
        x_pos = self.last_obs[0]
        y_pos = self.last_obs[1]
        if y_pos > -0.2:
            dist = self.model_inside.policy.get_distribution(obs_tensor)
        else:
            dist = self.model_outside.policy.get_distribution(obs_tensor)

        action = dist.sample().item()
        return action

env = create_env()


obs, _ = env.reset()
done = False


freeze_support()
model_1 = PPO.load("ppo_carracing1")
model_2 = PPO.load("ppo_carracing2")

env = SwitchingPolicyEnv(env, model_inside=model_2, model_outside=model_1)


obs, info = env.reset()
epizodes = 5000
total_reward = 0
ep_reward = 0
sucsess = 0
for _ in range(epizodes):
    total_reward +=ep_reward
    obs, info = env.reset()
    ep_reward = 0
    done = False
    truncated = False
    while not (done or truncated):
        action = env.get_policy_action()
        obs, reward, done, truncated, info = env.step(action)
        ep_reward += reward
        env.render()


        if done:
            if reward == 100:
                sucsess += 1
            print(f"Reward: {reward}, Done: {done}")
            print("Епізод завершено. Починаємо новий.")
            print(f"ep reward: {ep_reward}")
            break

env.close()
print(f"Загальний reward: {total_reward}")
print(f"середнє reward: {total_reward / epizodes}")
print(f"Загальний  success: {sucsess}")
