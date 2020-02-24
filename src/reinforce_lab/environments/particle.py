#!/usr/bin/env python

# Import modules
from dynamics import *

import numpy as np

from scipy.linalg import norm
from numpy.matlib import repmat


class Particle(Dynamics):
    alpha_ = 10.0
    beta_ = 10.0
    input_dim_ = 2
    state_dim_ = 2
    max_input_ = 20

    def __init__(self, num_envs=1, dyn_freq=1000):
        self.name_ = self.__class__.__name__

        Dynamics.__init__(self, num_envs, dyn_freq)

        # Create particle state [x y] (position & velocity same dimension)
        self.state_ = np.zeros((self.num_envs_, 2 * self.state_dim_))

        # init input vector [u]
        self.input_ = np.zeros((self.num_envs_, self.input_dim_))

    def Update(self):
        state_dot = (
            np.repeat(np.array([[self.alpha_, self.beta_]]), self.num_envs_, axis=0)
            * self.input_
        )

        self.state_[:, 0:2] += self.dt_ * state_dot
        self.state_[:, 2:4] = state_dot

        # print(self.state_.shape)
        # print(self.state_)

    def Reset(self, status):
        self.state_[status, 0:2] = np.random.uniform(
            low=-10, high=10, size=(np.sum(status), self.state_dim_)
        )
        self.state_[status, 2:4] = np.random.uniform(
            low=-1, high=1, size=(np.sum(status), self.state_dim_)
        )

    def Reward(self, desired_state):
        r, _ = self.rbf(self.state_[:, 0:2], desired_state, 1.0)

        return r[:, np.newaxis]

    def rbf(self, x, y, sigma=3.0):
        X = repmat(x, y.shape[0], 1)
        Y = y.repeat(x.shape[0], axis=0)

        k = np.exp(-norm(X - Y, axis=1) ** 2 / 2 / sigma ** 2)

        dk = (X - Y) / 2 / sigma ** 2 * k[:, np.newaxis]

        return k, dk

    # def SetInput(self, input):
    #     if np.any(input > self.max_input_):
    #         input[input > self.max_input_] = self.max_input_
    #     elif np.any(input < -self.max_input_):
    #         input[input < -self.max_input_] = -self.max_input_
    #     # if input > self.max_input_:
    #     #     input = self.max_input_
    #     # elif input < -self.max_input_:
    #     #     input = -self.max_input_
    #     # inp = [self.max_input_ if i > self.max_input_ else i for i in input]
    #     # inp = [-self.max_input_ if i < -self.max_input_ else i for i in inp]

    #     self.input_ = input

    # def Reward(self, desired_state):
    #     r = norm(self.state_ -
    #              desired_state.repeat(self.num_envs_, axis=0), axis=1)**2

    #     return r[:, np.newaxis]
