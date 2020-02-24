#!/usr/bin/env python
import sys

sys.path.insert(0, "../../reinforce_lab/")

import numpy as np

# Import modules
from environments.particle import *
from agents.actor_critic import *
from render_env import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

LOAD = True

# =============#
# ENVIRONMENT #
# =============#
env = Particle()
goal = np.array([3, 7])
limits = np.array([10, 10])

# =======#
# AGENT #
# =======#
if LOAD:
    agent = torch.load("agent.pt")
    agent.eval()
else:
    agent = ActorCritic(4, 2).to(device)

# ===========#
# ANIMATION #
# ===========#
fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(7, 6.5)
ax = plt.axes(xlim=(-10, 10), ylim=(-10, 10))

target = ax.plot(goal[0], goal[1], "r*", markersize=10)

anim = RenderEnv(env, agent, target, limits, device, fig, ax)

plt.show()
