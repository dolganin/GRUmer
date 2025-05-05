import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import numpy as np
from tqdm import trange
from dm_control import suite

class DMCWrapper:
    def __init__(self, domain, task):
        self.env = suite.load(domain, task)
        self.action_spec = self.env.action_spec()
        self.obs_dim = sum(np.prod(v.shape) for v in self.env.observation_spec().values())
        self.act_dim = self.action_spec.shape[0]
        self.reset()

    def reset(self):
        self.ts = self.env.reset()
        return self._flatten_obs(self.ts)

    def step(self, action):
        self.ts = self.env.step(action)
        obs = self._flatten_obs(self.ts)
        reward = self.ts.reward or 0.0
        done = self.ts.last()
        return obs, reward, done

    def _flatten_obs(self, ts):
        return np.concatenate([v.ravel() for v in ts.observation.values()])

class WorldModel(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, latent_dim, use_gru=True):
        super().__init__()
        self.use_gru = use_gru
        if use_gru:
            self.rnn = nn.GRU(obs_dim + act_dim, hidden_dim, batch_first=True)
            self.linear = nn.Linear(hidden_dim, latent_dim)
        else:
            self.fc = nn.Sequential(
                nn.Linear(obs_dim + act_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim)
            )

    def forward(self, obs, act, hidden=None):
        x = torch.cat([obs, act], dim=-1)
        if self.use_gru:
            x = x.unsqueeze(1)
            out, hidden = self.rnn(x, hidden)
            z = self.linear(out.squeeze(1))
        else:
            z = self.fc(x)
            hidden = None
        return z, hidden

class Actor(nn.Module):
    def __init__(self, latent_dim, act_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, act_dim)

    def forward(self, z):
        return torch.tanh(self.fc2(F.relu(self.fc1(z))))

class ReplayBuffer:
    def __init__(self, size=100000):
        self.buffer = deque(maxlen=size)

    def add(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        obs, act, _, next_obs, done = zip(*batch)
        return (
            torch.tensor(np.array(obs), dtype=torch.float32),
            torch.tensor(np.array(act), dtype=torch.float32),
            torch.tensor(np.array(next_obs), dtype=torch.float32),
            torch.tensor(np.array(done), dtype=torch.float32).unsqueeze(-1)
        )

def train_plan2explore(config):
    env = DMCWrapper("cheetah", "run")
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    wm1 = WorldModel(obs_dim, act_dim, config["hidden_dim"], config["latent_dim"], config["use_gru"])
    wm2 = WorldModel(obs_dim, act_dim, config["hidden_dim"], config["latent_dim"], config["use_gru"])
    actor = Actor(config["latent_dim"], act_dim, config["hidden_dim"])

    wm1_opt = optim.Adam(wm1.parameters(), lr=config["learning_rate"])
    wm2_opt = optim.Adam(wm2.parameters(), lr=config["learning_rate"])
    actor_opt = optim.Adam(actor.parameters(), lr=config["learning_rate"])

    buffer = ReplayBuffer(size=config["replay_buffer_size"])
    rewards = []

    for ep in trange(config["episodes"], desc="Plan2Explore"):
        obs = env.reset()
        total = 0
        hidden1 = hidden2 = None

        for _ in range(config["steps_per_episode"]):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            dummy_act = torch.zeros((1, act_dim))
            z, hidden1 = wm1(obs_t, dummy_act, hidden1)
            action = actor(z).squeeze(0).detach().numpy()

            next_obs, _, done = env.step(action)
            buffer.add(obs, action, 0.0, next_obs, done)
            obs = next_obs

            if len(buffer.buffer) >= config["batch_size"]:
                o, a, o2, d = buffer.sample(config["batch_size"])
                z1, _ = wm1(o, a)
                z2, _ = wm2(o, a)

                wm1_loss = F.mse_loss(z1, z1.detach())
                wm2_loss = F.mse_loss(z2, z2.detach())
                wm1_opt.zero_grad(); wm1_loss.backward(retain_graph=True); wm1_opt.step()
                wm2_opt.zero_grad(); wm2_loss.backward(retain_graph=True); wm2_opt.step()

                with torch.no_grad():
                    z_actor_input, _ = wm1(o, a)
                pred_action = actor(z_actor_input)
                z1_pred, _ = wm1(o, pred_action)
                z2_pred, _ = wm2(o, pred_action)
                std_pred = torch.std(torch.stack([z1_pred, z2_pred]), dim=0)
                intrinsic_reward = std_pred.mean(dim=1, keepdim=True)

                actor_loss = -intrinsic_reward.mean()
                actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()
                total += intrinsic_reward.sum().item()

            if done:
                break
        rewards.append(total)
    return rewards
