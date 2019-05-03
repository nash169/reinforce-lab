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
from unicycle import *
from ppo import *
from actor_critic import *


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

LOAD = True

#=============#
# ENVIRONMENT #
#=============#

dyn_freq = 1000  # hz - the trajectory parametrization frequency should match this one
T = 10  # s - It supposed to be the final time but let's see
t = 0

# Initial state
state_0 = State(3, 3)
limits = np.array([10, 10])

# Create the environment and set the initial postion/velocity
env = Unicycle(dyn_freq)
env.SetState(state_0)

# Curve y = sin(x) - simple parmetrization
t_param = np.arange(0, T, 1/dyn_freq)[:, np.newaxis]
x_ref = t_param - 5
y_ref = np.sin(t_param)
traj = np.concatenate((x_ref, y_ref), axis=1)

reward = env.Reward(traj[0, :][:, np.newaxis])

#=======#
# AGENT #
#=======#
control_freq = 100  # hz
sensor_freq = 0  # hz

# Create Actor-Critic
if LOAD:
    agent = torch.load('storage/agent.pt')
    agent.eval()
else:
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0., std=0.1)
            nn.init.constant_(m.bias, 0.1)

    agent = ActorCritic(6, 1, 20).to(device)

# Initial agent outputs
action = np.array([0])[:, np.newaxis]
dist, value = agent(torch.FloatTensor(np.concatenate(
    (state_0.position, state_0.velocity)).transpose()).to(device))
likelihood = dist.log_prob(torch.FloatTensor(action.transpose()).to(device))
# action_lims = np.array([-1, 1]) Consider the option of bounding the action

#===========#
# OPTIMIZER #
#===========#
update_freq = 10
epoch = 1
max_epochs = 10
batch_size = int(1000/update_freq)  # Better to call it num_steps
mini_batch_size = int(batch_size/10)
ppo_epochs = 4
entropy = 0
lr = 3e-4
optimizer = optim.Adam(agent.parameters(), lr=lr)

#========#
# RECORD #
#========#
state_log = np.concatenate((state_0.position, state_0.velocity)).transpose()
reward_log = np.array([reward])[:, np.newaxis].transpose()
mask_log = np.array([1])[:, np.newaxis].transpose()
dist_log = np.array([dist])[:, np.newaxis].transpose()
likelihood_log = likelihood.cpu().detach().numpy()
action_log = action.transpose()
value_log = value.cpu().detach().numpy()

reward_mean = []

epoch_size = 0

while epoch <= max_epochs:
    # Control loop running at 100hz
    if not (t+1) % (1000/control_freq) and t:
        # Simulate the actuator dealy
        action = dist_log[int(t-1000/control_freq)].sample()
        action = action.cpu().numpy()

    # Reset the simulation if the epoch ends
    if not (t+1) % (T*dyn_freq) or np.any(np.absolute(state_log[-1][0:2]) >= limits):
        print('End epoch '+str(epoch)+' - Start epoch '+str(epoch+1))
        env.Reset()
        t = 0
        epoch += 1
        action = np.array([0])[:, np.newaxis]
        reward_mean = np.append(reward_mean, np.sum(
            reward_log[epoch_size:state_log.shape[0]])/(state_log.shape[0]-epoch_size))
        epoch_size = state_log.shape[0]
    else:
        # Dynamics loop running at 1000hz - t -> t+1
        env.SetInput(action)
        env.Update()
        t += 1  # Increment time

    state = env.GetState()
    reward = env.Reward(traj[t, :])

    # Query the agent at same freq of the dynamics integration -> Sensor frequency = Dynamics frequency
    dist, value = agent(torch.FloatTensor(np.concatenate((state.position, state.velocity)).transpose()).to(
        device))  # This compute the policy and the value function at step t+1
    # Check here - the action gets updated at a lower frequency
    likelihood = dist.log_prob(torch.FloatTensor(action).to(device))
    entropy += dist.entropy().mean()

    # Record step t+1
    state_log = np.append(state_log, np.concatenate(
        (state.position, state.velocity)).transpose(), axis=0)
    reward_log = np.append(reward_log, np.array([reward])[
                           :, np.newaxis].transpose(), axis=0)
    value_log = np.append(value_log, value.cpu().detach().numpy(), axis=0)
    action_log = np.append(action_log, action.transpose(), axis=0)
    dist_log = np.append(dist_log, np.array([dist])[:, np.newaxis].transpose())
    likelihood_log = np.append(
        likelihood_log, likelihood.cpu().detach().numpy(), axis=0)

    if not t % (T*dyn_freq) and t:
        mask_log = np.append(mask_log, np.array(
            [0])[:, np.newaxis].transpose(), axis=0)
    else:
        mask_log = np.append(mask_log, np.array(
            [1])[:, np.newaxis].transpose(), axis=0)

    # Update the policy
    if not (t+1) % (1000/update_freq) and t:
        returns = compute_gae(value.cpu().detach().numpy(),
                              reward_log[int(-batch_size):-1],
                              mask_log[int(-batch_size):-1],
                              value_log[int(-batch_size):-1])

        advantage = returns - value_log[int(-batch_size):-1]

        ppo_update(agent,
                   optimizer,
                   ppo_epochs,
                   mini_batch_size,
                   torch.FloatTensor(
                       state_log[int(-batch_size):-1]).to(device),
                   torch.FloatTensor(
                       action_log[int(-batch_size):-1]).to(device),
                   torch.FloatTensor(
                       likelihood_log[int(-batch_size):-1]).to(device),
                   torch.FloatTensor(returns).to(device),
                   torch.FloatTensor(advantage).to(device))

        entropy = 0

torch.save(agent, 'storage/agent.pt')

np.save('storage/reward.npy', reward_mean)
np.save('storage/epochs.npy', np.arange(1, epoch))

plt.plot(np.arange(1, epoch), reward_mean)

#===========#
# ANIMATION #
#===========#
# fig = plt.figure()
# fig.set_dpi(100)
# fig.set_size_inches(7, 6.5)
# ax = plt.axes(xlim=(-10, 10), ylim=(-10, 10))

# patch = plt.Circle((5, 5), 0.5, fc='y')
# line, = ax.plot([], [], '--b', label='trajectory', lw=2)
# line_ref, = ax.plot(traj[:,0],traj[:,1],'r', label='reference')
