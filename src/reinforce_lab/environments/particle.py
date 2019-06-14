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
        # self.state_ = State(2, 2)
        self.state_ = np.zeros((self.num_envs_, self.state_dim_))

        # init input vector [u]
        self.input_ = np.zeros((self.num_envs_, self.input_dim_))

    def Update(self):
        self.state_ += self.dt_ * \
            np.repeat(np.array([[self.alpha_, self.beta_]]),
                      self.num_envs_, axis=0)*self.input_

    def Reset(self, status=True):
        self.state_[status, :] = np.random.uniform(
            low=-10, high=10, size=(np.sum(status), self.state_dim_))

    # def Reward(self, desired_state):
    #     r = norm(self.state_ -
    #              desired_state.repeat(self.num_envs_, axis=0), axis=1)**2

    #     return r[:, np.newaxis]

    def Reward(self, desired_state):
        r, _ = rbf(self.state_, desired_state)

        return r[:, np.newaxis]
