#!/usr/bin/env python
import sys
sys.path.insert(0, '../../reinforce_lab/')

import matplotlib.pyplot as plt

from agents.actor_critic import *
from optimizers.ppo import ppo_update, compute_gae
from environments.particle import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

LOAD = False

#=============#
# ENVIRONMENT #
#=============#
# Number of environments
num_envs = 50
# hz - dynamics integration frequency (the trajectory parametrization frequency should match this one)
dyn_freq = 1000
# s - Better defined as the maximum time to reach the goal (in trajectory tracking needs more explanation)
T = 10
# Physical environment boundaries
limits = np.array([10, 10])
# Time vector
t = np.zeros(num_envs)
# Create the environment
env = Particle(num_envs)
# Initial state
state = np.zeros((num_envs, 4))
# Set the initial postion (velocity is not needed if there is no dynamics)
env.SetState(state)
# Goal of the particle
target = np.array([[3, 7]])

#=======#
# AGENT #
#=======#
# hz - Control/Actuator frequecy
control_freq = 1000
# hz - Sensor frequency
sensor_freq = 0

if LOAD:
    # Load the Actor-Critic agent
    agent = torch.load('agent.pt')
    agent.eval()
else:
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0., std=0.1)
            nn.init.constant_(m.bias, 0.1)

    # Actor network with 2 inputs (position) and 2 outputs (particle trust)
    agent = ActorCritic(4, 2, 20).to(device)


# Initial agent outputs
action = np.zeros((num_envs, 2))
next_action = np.zeros((num_envs, 2))

#===========#
# OPTIMIZER #
#===========#
# Total number of epochs
epochs = 500
# Number of steps before calling the optimizer
num_steps = 200
# Size of the minibatch
mini_batch_size = 20
# Number of epochs for which running the optimizer
ppo_epochs = 5
# Learning rate
lr = 3e-6
# Set optimizer to Adam
optimizer = optim.Adam(agent.parameters(), lr=lr)

#========#
# RECORD #
#========#
states = []
rewards = []
reward_means = []
masks = []
log_policies = []
actions = []
values = []
entropy = 0
actor_losses = []
value_losses = []

for epoch in range(epochs):
    print("Epoch: ", epoch)
    for _ in range(num_steps+1):
        # Query the agent at same freq of the dynamics integration -> Sensor frequency = Dynamics frequency
        # This compute the policy and the value function at step t+1
        dist, value = agent(torch.FloatTensor(state).to(device))

        # Get the log_policy - Check if it is ok to reduce the frequency here
        log_policy = dist.log_prob(torch.FloatTensor(action).to(device))
        entropy += dist.entropy().mean()

        # Control loop running at 100hz
        act_to_update = t % (dyn_freq/control_freq) == 0
        action[act_to_update, :] = next_action[act_to_update, :]
        # Simulate the actuator dealy
        next_action[act_to_update, :] = dist.sample().cpu().numpy()[
            act_to_update, :]

        # Get state and reward
        state = env.GetState()
        reward = env.Reward(target)

        # Calculate the ending states
        state_to_reset = (t % (T*dyn_freq) == 0)*(t != 0) + \
            np.any(np.absolute(state[:,0:2]) >= limits, axis=1)
        mask = state_to_reset*1

        # Record step t
        states.append(torch.FloatTensor(state).to(device))
        rewards.append(torch.FloatTensor(reward).to(device))
        values.append(value)
        actions.append(torch.FloatTensor(action).to(device))
        log_policies.append(log_policy)
        masks.append(torch.FloatTensor(mask[:, np.newaxis]).to(device))

        # Dynamics loop running at 1000hz - t -> t+1
        env.SetInput(action)
        env.Update()
        t += 1  # Increment time

        # Reset the simulation if the epoch ends or agent is out of limits
        env.Reset(state_to_reset)
        t[state_to_reset] = 0
        action[state_to_reset] = np.zeros((1, 2))
        next_action[state_to_reset, :] = np.zeros((1, 2))

    # Update the policy
    returns = compute_gae(value, rewards[-int(num_steps+1):-1],
                          masks[-int(num_steps+1):-1], values[-int(num_steps+1):-1])

    advantage = torch.cat(returns).detach() - \
        torch.cat(values[-int(num_steps+1):-1]).detach()

    actor_loss, value_loss = ppo_update(agent, optimizer, ppo_epochs, mini_batch_size,
                                        torch.cat(
                                            states[-int(num_steps+1):-1]).detach(),
                                        torch.cat(
                                            actions[-int(num_steps+1):-1]).detach(),
                                        torch.cat(
                                            log_policies[-int(num_steps+1):-1]).detach(),
                                        torch.cat(returns).detach(),
                                        advantage)

    actor_losses.append(actor_loss.cpu().detach().numpy())
    value_losses.append(value_loss.cpu().detach().numpy())
    reward_means.append(
        np.mean(torch.cat(rewards[-int(num_steps+1):]).cpu().detach().numpy()))
    entropy = 0


torch.save(agent, 'agent.pt')
np.save('actor_losses.npy', actor_losses)

fig = plt.figure()
ax = fig.gca()
ax.plot(range(len(actor_losses)), actor_losses)
plt.title("Actor Loss")

fig = plt.figure()
ax = fig.gca()
ax.plot(range(len(reward_means)), reward_means)
plt.title("Mean reward per epoch")

plt.show()
