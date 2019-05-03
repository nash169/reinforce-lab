#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np


class RenderEnv(animation.FuncAnimation):
    t_ = 0
    T_ = 10
    control_freq_ = 100
    dyn_freq_ = 1000
    epoch_size_ = 0

    def __init__(self, env, agent, target, device, fig=None, ax=None, frames=None, interval=1, blit=True):
        if fig is None:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.get_figure()
        else:
            if ax is None:
                ax = fig.gca()

        self.fig_ = fig
        self.ax_ = ax

        self.device_ = device

        self.env_ = env
        state_0 = self.env_.GetState()
        self.state_log_ = np.concatenate(
            (state_0.position, state_0.velocity)).transpose()
        self.limits_ = np.array([10, 10])

        self.agent_ = agent
        dist, _ = agent(torch.FloatTensor(np.concatenate((state_0.position, state_0.velocity)).transpose()).to(self.device_))
        self.dist_log_ = np.array([dist])[:, np.newaxis].transpose()
        self.action_ = np.array([0, 0])[:, np.newaxis]

        self.patch_ = plt.Circle((5, 5), 0.5, fc='y')
        self.traj_, = ax.plot([], [], '--b', label='trajectory', lw=2)
        self.goal_, = target

        super(RenderEnv, self).__init__(fig, self.animate, init_func=self.init_anim,
                                                  frames=frames, interval=interval, blit=blit)

    def init_anim(self):
        self.patch_.center = (5, 5)
        self.ax_.add_patch(self.patch_)
        self.traj_.set_data([], [])

        return self.patch_, self.traj_, self.goal_

    def animate(self, i):
        # Control loop running at 100hz
        if not (self.t_+1) % (1000/self.control_freq_) and self.t_:
            # Simulate the actuator dealy
            self.action_ = self.dist_log_[int(self.t_-1000/self.control_freq_)].sample()
            self.action_ = self.action_.cpu().numpy().transpose()

        # Reset the simulation if the epoch ends
        if not (self.t_+1) % (self.T_*self.dyn_freq_) or np.any(np.absolute(self.state_log_[-1][0:2]) >= self.limits_):
            self.env_.Reset()
            self.t_ = 0
            self.action_ = np.array([0, 0])[:, np.newaxis]
            self.epoch_size_ = self.state_log_.shape[0]
        else:
            # Dynamics loop running at 1000hz - t -> t+1
            self.env_.SetInput(self.action_)
            self.env_.Update()
            self.t_ += 1  # Increment time

        state = self.env_.GetState()

        # Query the agent at same freq of the dynamics integration -> Sensor frequency = Dynamics frequency
        dist, _ = self.agent_(torch.FloatTensor(np.concatenate((state.position, state.velocity)).transpose()).to(
            self.device_))  # This compute the policy and the value function at step t+1

        self.state_log_ = np.append(self.state_log_, np.concatenate(
            (state.position, state.velocity)).transpose(), axis=0)
        self.dist_log_ = np.append(self.dist_log_, np.array([dist])[:, np.newaxis].transpose())

        self.patch_.center = (self.state_log_[-1, 0], self.state_log_[-1, 1])
        self.traj_.set_data(self.state_log_[self.epoch_size_:, 0],
                            self.state_log_[self.epoch_size_:, 1])

        return self.patch_, self.traj_, self.goal_
