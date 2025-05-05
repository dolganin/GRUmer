from stable_baselines3 import PPO, TD3
from stable_baselines3.common.env_util import make_vec_env
from gymnasium import Env, spaces
from dm_control import suite
import numpy as np

class DMCGymWrapper(Env):
    def __init__(self, domain, task):
        super().__init__()
        self.env = suite.load(domain, task)
        self.ts = self.env.reset()
        self.obs_dim = sum(v.size for v in self.env.observation_spec().values())
        self.act_dim = self.env.action_spec().shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.act_dim,), dtype=np.float32)

    def _flatten_obs(self, ts):
        return np.concatenate([v.ravel() for v in ts.observation.values()])

    def reset(self, seed=None, options=None):
        self.ts = self.env.reset()
        return self._flatten_obs(self.ts), {}

    def step(self, action):
        self.ts = self.env.step(action)
        obs = self._flatten_obs(self.ts)
        reward = self.ts.reward or 0.0
        done = self.ts.last()
        return obs, reward, done, False, {}

def train_baselines(algo_name, save_dir):
    env = make_vec_env(lambda: DMCGymWrapper("cheetah", "run"), n_envs=1)
    algo = PPO if algo_name.lower() == "ppo" else TD3
    model = algo("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=30 * 500)
    rewards = []
    obs = env.reset()
    for _ in range(30):
        total = 0
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            total += reward[0]
        rewards.append(total)
        obs = env.reset()
    return rewards
