import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from torch import nn


class MetaFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int):
        super(MetaFeatureExtractor, self).__init__(observation_space, features_dim=1)
        obs_dim = observation_space.shape[0]
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self._features_dim = 256
        self.action_dim = action_dim

    def forward(self, obs):
        return self.fc(obs)


def create_meta_env(env_name, model1, model2):
    env = gym.make(env_name,continuous=False, gravity=-10.0,
                   enable_wind=True, wind_power=15.0, turbulence_power=1.5, render_mode="human")

    # Обгортаємо середовище для meta-агента
    class MetaEnv(gym.Wrapper):
        def __init__(self, env, model1, model2):
            super().__init__(env)
            self.model1 = model1
            self.model2 = model2

            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            low = np.concatenate([env.observation_space.low, np.zeros(2 * action_dim)])
            high = np.concatenate([env.observation_space.high, np.ones(2 * action_dim)])
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                dtype=np.float32,
                shape=(obs_dim + 2 * action_dim,)
            )

        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            obs = self._augment_obs(obs)
            return obs, info

        def step(self, action):
            obs, reward, done, truncated, info = self.env.step(action)
            obs = self._augment_obs(obs)
            return obs, reward, done, truncated, info

        def _augment_obs(self, obs):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            dist1 = self.model1.policy.get_distribution(obs_tensor)
            dist2 = self.model2.policy.get_distribution(obs_tensor)

            # Отримуємо ймовірності дій обох моделей
            probs1 = dist1.distribution.probs.squeeze(0).detach().numpy()
            probs2 = dist2.distribution.probs.squeeze(0).detach().numpy()

            # Конкатенуємо спостереження з повними ймовірностями
            augmented_obs = np.concatenate([obs, probs1, probs2]).astype(np.float32)

            # Оновлюємо shape вручну, якщо потрібно:
            if self.observation_space.shape[0] != augmented_obs.shape[0]:
                self.observation_space = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(augmented_obs.shape[0],),
                    dtype=np.float32
                )

            return augmented_obs

    return MetaEnv(env, model1, model2)

class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.last_percent = -1

    def _on_step(self) -> bool:
        percent = int((self.num_timesteps / self.total_timesteps) * 100)
        if percent > self.last_percent:
            self.last_percent = percent
            print(f"🚀 Training progress: {percent}%")
        return True

def create_env(a=False):
    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                   enable_wind=True, wind_power=15.0, turbulence_power=1.5,render_mode="human" if a else None)
    return env

def train_agent(name: str = "ppo_carracing"):
    env = SubprocVecEnv([create_env for _ in range(6)])
    model = PPO("MlpPolicy", env, learning_rate=3e-4,n_steps=2048,batch_size=256,gamma=0.99,gae_lambda=0.95,clip_range=0.2,ent_coef=0, policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),)
    total_timesteps = 20000000
    progress_callback = ProgressCallback(total_timesteps=total_timesteps)

    model.learn(total_timesteps=total_timesteps,callback=progress_callback)

    model.save(name)

    return model

def load_agent():
    model = PPO.load("ppo_carracing1")
    return model

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  # для Windows
    #model = train_agent("ppo_carracing2")
    model_1 = PPO.load("ppo_carracing1")
    model_2 = PPO.load("ppo_carracing2")


    def make_meta_env():
        return create_meta_env("LunarLander-v3", model_1, model_2)
    env = make_meta_env()

    vec_env = make_vec_env(make_meta_env, n_envs=4)

    action_dim = vec_env.action_space.n

    policy_kwargs = dict(
        features_extractor_class=MetaFeatureExtractor,
        features_extractor_kwargs=dict(action_dim=action_dim),
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
    )

    # meta_model = PPO(
    #     "MlpPolicy",
    #     vec_env,
    #     policy_kwargs=policy_kwargs,
    #     learning_rate=3e-4,
    #     batch_size=256,
    #     n_steps=2048,
    #     gamma=0.99,
    #     gae_lambda=0.95,
    #     clip_range=0.2,
    #     ent_coef=0,
    #     verbose=1,
    #     tensorboard_log="./ppo_tensorboard/"
    # )

    # meta_model.learn(total_timesteps=3_000_000, tb_log_name="PPO_run_1")
    #
    # meta_model.save("meta_agent")
    meta_model = PPO.load("meta_agent")

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
            action, _ = meta_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward


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


