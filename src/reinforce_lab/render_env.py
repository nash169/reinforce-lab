#!/usr/bin/env python

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation


class RenderEnv(animation.FuncAnimation):
    T_ = 10
    control_freq_ = 100
    dyn_freq_ = 1000
    epoch_size_ = 0

    def __init__(self, env, agent, target, limits, device, fig=None, ax=None, frames=None, interval=1, blit=True):
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
        self.agent_ = agent
        self.goal_, = target
        self.limits_ = limits

        self.t_ = np.zeros(1)
        self.env_.Reset()
        self.state_ = self.env_.GetState()
        self.action_ = np.zeros((1, self.env_.input_dim_))
        self.next_action_ = np.zeros((1, self.env_.input_dim_))

        self.state_log_ = np.array([]).reshape(0, self.env_.state_dim_)

        self.patch_ = plt.Circle((5, 5), 0.5, fc='y')
        self.traj_, = ax.plot([], [], '--b', label='trajectory', lw=2)

        super(RenderEnv, self).__init__(fig, self.animate, init_func=self.init_anim,
                                        frames=frames, interval=interval, blit=blit)

    def init_anim(self):
        self.patch_.center = (5, 5)
        self.ax_.add_patch(self.patch_)
        self.traj_.set_data([], [])

        return self.patch_, self.traj_, self.goal_

    def animate(self, i):
        # Query the agent at same freq of the dynamics integration -> Sensor frequency = Dynamics frequency
        # This compute the policy and the value function at step t+1
        dist, _ = self.agent_(torch.FloatTensor(self.state_).to(self.device_))

        # Control loop running at 100hz
        act_to_update = self.t_ % (self.dyn_freq_/self.control_freq_) == 0
        self.action_[act_to_update, :] = self.next_action_[act_to_update, :]
        self.next_action_[act_to_update, :] = dist.sample().cpu().numpy()[
            act_to_update, :]

        self.state_ = self.env_.GetState()
        state_to_reset = (self.t_ % (self.T_*self.dyn_freq_) == 0)*(self.t_ != 0) + \
            np.any(np.absolute(self.state_) >= self.limits_, axis=1)

        # Record step t
        self.state_log_ = np.append(self.state_log_, self.state_, axis=0)

        # Dynamics loop running at 1000hz - t -> t+1
        self.env_.SetInput(self.action_)
        self.env_.Update()
        self.t_ += 1  # Increment time

        # Reset the simulation if the epoch ends or agent is out of limits
        self.env_.Reset(state_to_reset)
        self.t_[state_to_reset] = 0
        self.action_[state_to_reset] = np.zeros((1, self.env_.input_dim_))
        self.next_action_[state_to_reset, :] = np.zeros(
            (1, self.env_.input_dim_))

        self.patch_.center = (self.state_log_[-1, 0], self.state_log_[-1, 1])
        self.traj_.set_data(self.state_log_[:, 0], self.state_log_[:, 1])

        if state_to_reset:
            self.state_log_ = np.array([]).reshape(0, self.env_.state_dim_)

        return self.patch_, self.traj_, self.goal_
