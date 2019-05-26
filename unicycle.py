#!/usr/bin/env python

# Import modules
from dynamics import *


class Unicycle(Dynamics):
    # Model variables
    L_ = 2.       # Vehicle length
    v_0_ = 1.     # Vehicle speed
    input_dim_ = 1
    state_dim_ = 3

    def __init__(self, dyn_freq=1000, num_envs=1):
        self.name_ = self.__class__.__name__

        Dynamics.__init__(self, dyn_freq, num_envs)

        # Create unicycle state [x y theta] (position & velocity same dimension)
        # Velocity not considered at this stage because it still a kinematics
        self.state_ = np.zeros(self.num_envs_, self.state_dim_)

        # init input vector [u]
        self.input_ = np.zeros(self.num_envs_, self.input_dim_)

    def Update(self):
        self.state_[:, 0] += self.v_0_*np.cos(self.state_[:, 2])
        self.state_[:, 1] += self.v_0_*np.sin(self.state_[:, 2])
        self.state_[:, 2] += self.v_0_/self.L_*np.tan(self.input_)

    def Reset(self, status=True):
        self.state_[status, :-1] = np.random.uniform(
            low=-10, high=10, size=(np.sum(status), self.state_dim_-1))
        self.state_[
            status, -1] = np.random.uniform(low=0, high=2*np.pi, size=(np.sum(status), 1))

    def Reward(self, desired_state):
        r = norm(self.state_[:, :-1] -
                 desired_state.repeat(self.num_envs_, axis=0), axis=1)**2

        return r[:, np.newaxis]
