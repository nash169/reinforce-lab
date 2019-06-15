#!/usr/bin/env python

# Import modules
from dynamics import *
from kernels import rbf


class Particle(Dynamics):
    alpha_ = 5.
    beta_ = 5.
    input_dim_ = 2
    state_dim_ = 2

    def __init__(self, num_envs=1, dyn_freq=1000):
        self.name_ = self.__class__.__name__

        Dynamics.__init__(self, num_envs, dyn_freq)

        # Create particle state [x y] (position & velocity same dimension)
        self.state_ = np.zeros((self.num_envs_, 2*self.state_dim_))

        # init input vector [u]
        self.input_ = np.zeros((self.num_envs_, self.input_dim_))

    def Update(self):
        state_dot = np.repeat(
            np.array([[self.alpha_, self.beta_]]), self.num_envs_, axis=0)*self.input_
        self.state_[:, 0:2] += self.dt_*state_dot
        self.state_[:, 2:4] += state_dot

    def Reset(self, status):
        self.state_[status, 0:2] = np.random.uniform(
            low=-10, high=10, size=(np.sum(status), self.state_dim_))
        self.state_[status, 2:4] = np.random.uniform(
            low=-1, high=1, size=(np.sum(status), self.state_dim_))

    def Reward(self, desired_state):
        r, _ = rbf(self.state_[:, 0:2], desired_state)

        return r[:, np.newaxis]

    # def Reward(self, desired_state):
    #     r = norm(self.state_ -
    #              desired_state.repeat(self.num_envs_, axis=0), axis=1)**2

    #     return r[:, np.newaxis]
