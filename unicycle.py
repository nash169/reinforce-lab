#!/usr/bin/env python

# Import modules
import numpy as np
import math as mt

# Import classes
from state import *
from dynamics import *

class Unicycle(Dynamics):
    # Model variables
    L_ = 2       # Vehicle length
    v_0_ = 1     # Vehicle speed

    def __init__(self, dyn_freq=1000):
        self.name_ = self.__class__.__name__

        Dynamics.__init__(self, dyn_freq)
        
        # Create unicycle state [x y theta] (position & velocity same dimension)
        self.state_ = State(3,3)
        
        # init input vector [u]
        self.input_ = np.zeros(1)

    def Update(self):
        self.state_.velocity[0] = self.v_0_*mt.cos(self.state_.position[2])
        self.state_.velocity[1] = self.v_0_*mt.sin(self.state_.position[2])
        self.state_.velocity[2] = self.v_0_/self.L_*mt.tan(self.input_[0])

        self.state_.position = self.state_.position + self.dt_*self.state_.velocity
    
    def Reset(self):
        self.state_.position[:-1] = np.random.uniform(low=-10,high=10,size=(2,1))
        self.state_.position[-1] = np.random.uniform(low=0,high=2*mt.pi,size=(1,1))
        self.state_.velocity = np.zeros((3,1))

    def Reward(self, desired_state):
        r = mt.pow(np.linalg.norm(self.state_.postion[:-1] - desired_state), 2)

        return r