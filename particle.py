#!/usr/bin/env python

# Import modules
import numpy as np
import math as mt

# Import classes
from state import *
from dynamics import *


class Particle(Dynamics):
    alpha_ = 5

    def __init__(self, dyn_freq=1000):
        self.name_ = self.__class__.__name__

        Dynamics.__init__(self, dyn_freq)

        # Create particle state [x y] (position & velocity same dimension)
        self.state_ = State(2, 2)

        # init input vector [u]
        self.input_ = np.zeros(2)[:, np.newaxis]

    def Update(self):
        self.state_.velocity[0] = self.alpha_*self.input_[0]
        self.state_.velocity[1] = self.alpha_*self.input_[1]

        self.state_.position = self.state_.position + self.dt_*self.state_.velocity

    def Reset(self):
        self.state_.position = np.random.uniform(
            low=-10, high=10, size=(self.state_.position.size, 1))
        self.state_.velocity = np.zeros((self.state_.velocity.size, 1))

    def Reward(self, desired_state):
        r = mt.pow(np.linalg.norm(self.state_.position - desired_state), 2)

        return r
