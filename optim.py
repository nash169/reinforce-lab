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

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

LOAD = False

#=============#
# ENVIRONMENT #
#=============#

dyn_freq = 1000  # hz - the trajectory parametrization frequency should match this one
T = 10  # s - It supposed to be the final time but let's see
t = 0

# Initial state
state_0 = State(2, 2)
limits = np.array([10, 10])

# Create the environment and set the initial postion/velocity
env = Particle(dyn_freq)
env.SetState(state_0)

# Curve y = sin(x) - simple parmetrization
target = np.array([3, 7])
reward = env.Reward(target)

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

    agent = ActorCritic(4, 2, 20).to(device)

# Initial agent outputs
action = np.array([0, 0])[:, np.newaxis]
dist, value = agent(torch.FloatTensor(np.concatenate(
    (state_0.position, state_0.velocity)).transpose()).to(device))
dist_act = dist
likelihood = dist.log_prob(torch.FloatTensor(action.transpose()).to(device))
# action_lims = np.array([-1, 1]) Consider the option of bounding the action

#===========#
# OPTIMIZER #
#===========#
update_freq = 10
epoch = 1
max_epochs = 200
batch_size = int(1000/update_freq)  # Better to call it num_steps
mini_batch_size = int(batch_size/10)
ppo_epochs = 4
entropy = 0
lr = 3e-6
optimizer = optim.Adam(agent.parameters(), lr=lr)

#========#
# RECORD #
#========#
state_log = np.concatenate((state_0.position, state_0.velocity)).transpose()
reward_log = np.array([reward])[:, np.newaxis].transpose()
mask_log = np.array([1])[:, np.newaxis].transpose()
likelihood_log = likelihood.cpu().detach().numpy()
action_log = action.transpose()
value_log = value.cpu().detach().numpy()

reward_mean = []

epoch_size = 0

while epoch <= max_epochs:
    # Control loop running at 100hz
    if not t % (1000/control_freq) and t:
        # Simulate the actuator dealy
        action = dist_act.sample()
        action = action.cpu().numpy().transpose()
        dist_act = dist

    # Reset the simulation if the epoch ends
    if not (t+1) % (T*dyn_freq) or np.any(np.absolute(state_log[-1][0:2]) >= limits):
        print('End epoch '+str(epoch)+' - Start epoch '+str(epoch+1))
        env.Reset()
        t = 0
        epoch += 1
        action = np.array([0, 0])[:, np.newaxis]
        reward_mean = np.append(reward_mean, np.sum(
            reward_log[epoch_size:state_log.shape[0]])/(state_log.shape[0]-epoch_size))
        epoch_size = state_log.shape[0]
    else:
        # Dynamics loop running at 1000hz - t -> t+1
        env.SetInput(action)
        env.Update()
        t += 1  # Increment time

    state = env.GetState()
    if math.isnan(state.position[0]):
        break

    reward = env.Reward(target)

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
    likelihood_log = np.append(
        likelihood_log, likelihood.cpu().detach().numpy(), axis=0)

    if not t % (T*dyn_freq) and t:
        mask_log = np.append(mask_log, np.array(
            [0])[:, np.newaxis].transpose(), axis=0)
    else:
        mask_log = np.append(mask_log, np.array(
            [1])[:, np.newaxis].transpose(), axis=0)

    # Update the policy
    if not (t+1) % (1000/update_freq):
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