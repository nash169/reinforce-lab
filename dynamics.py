#!/usr/bin/env python

# Import modules
import numpy as np
from scipy.linalg import norm


class Dynamics:

    def __init__(self, num_envs=1, dyn_freq=1000):
        # Set time step
        self.dt_ = 1/dyn_freq
        self.num_envs_ = num_envs

    def SetState(self, state):
        self.state_ = state

    def SetInput(self, input):
        self.input_ = input

    def GetState(self):
        return self.state_

    def GetInput(self):
        return self.input_

    def __getitem__(self, key):
        return self.__getattribute__[key]

    def __setitem__(self, key, value):
        self.__setattr__[key] = value
