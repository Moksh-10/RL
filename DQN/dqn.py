import torch
from torch import nn
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
import cv2
# from IPython import display
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make('ALE/Breakout-v5', render_mode="human")
# print('action space', env.action_space)
# print('obs space', env.observation_space)

# obs = env.reset()
# print(len(obs))

class dqn(nn.Module):
    def __init__(self, n_acts):
        super().__init__()

        self.l1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU()
        )

        self.l3 = nn.Sequential(
            nn.Linear(32*12*9, 256),
            nn.ReLU(),
            nn.Linear(256, n_acts)
        )


    def forward(self, obs):
        q_vals = self.l1(obs)
        q_vals = q_vals.view(-1, 32*12*9)
        q_vals = self.l3(q_vals)
        return q_vals

dqn = dqn(n_acts=4)
gg = dqn(torch.zeros(1, 3, 110, 84))
print(gg)

# max_steps = 1000
# obs = env.reset()
# for step in range(max_steps):
#     act = random.randint(0, 3)
#     obs, reward, ter, trun, info = env.step(act)
#
#     env.render()
#     time.sleep(0.05)
#
#     if ter or trun:
#         break
#         # obs, info = env.reset()





