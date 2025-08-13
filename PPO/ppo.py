import torch
from torch import nn, optim
from torch.distributions.categorical import Categorical
import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class actor_critic(nn.Module):
    def __init__(self, obs_space_size, act_space_size):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_space_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.policy = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_space_size)
        )

        self.value = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def value(self, obs):
        z = self.shared(obs)
        val = self.value(z)
        return val

    def policy(self, obs):
        z = self.shared(obs)
        pol = self.policy(z)
        return pol

    def forward(self, obs):
        z = self.shared(obs)
        pol = self.policy(z)
        val = self.value(z)
        return pol, val


def rollout(model, env, max_steps = 1000):
    train_data = []
    obs = env.reset()

    ep_reward = 0
    for _ in range(max_steps):
        logits, val = model(torch.tensor([obs], dtype=torch.float32))
        act_distr = Categorical(logits)
        act = act_distr.sample()
        act_log_prob = act_distr.log_prob(act).item()

        next_obs, reward, done, _ = env.step(act.item())

        obs = next_obs
        ep_reward += reward
        if done:
            break

    return train_data, ep_reward


