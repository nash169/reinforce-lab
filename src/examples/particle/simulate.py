#!/usr/bin/env python
import sys
sys.path.insert(0, '../reinforce_lab/')

# Import modules
from particle import *
from actor_critic import *
from render_env import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

LOAD = True

#=============#
# ENVIRONMENT #
#=============#
env = Particle()
goal = np.array([3, 7])
limits = np.array([10, 10])

#=======#
# AGENT #
#=======#
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

target = ax.plot(goal[0], goal[1], 'r*', markersize=10)

anim = RenderEnv(env, agent, target, limits, device, fig, ax)

plt.show()
