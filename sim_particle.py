#!/usr/bin/env python

# Import modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

# Import classes
from particle import *
from ppo import *
from actor_critic import *
from render_env import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

LOAD = True

#=============#
# ENVIRONMENT #
#=============#

# Create the environment and set the initial postion/velocity
env = Particle(1000)
env.SetState(State(2, 2))

# Curve y = sin(x) - simple parmetrization
target = np.array([3, 7])
reward = env.Reward(target)

#=======#
# AGENT #
#=======#

# Create Actor-Critic
if LOAD:
    agent = torch.load('storage/agent.pt')
    agent.eval()
else:
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0., std=0.1)
            nn.init.constant_(m.bias, 0.1)

    agent = ActorCritic(4, 2, 20).to(device)

#===========#
# ANIMATION #
#===========#
fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(7, 6.5)
ax = plt.axes(xlim=(-10, 10), ylim=(-10, 10))

target = ax.plot(target[0], target[1], 'r*', markersize=10)

anim = RenderEnv(env, agent, target, device, fig, ax)

plt.show()