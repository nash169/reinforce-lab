#!/usr/bin/env python

# Import modules
import numpy as np
import math as mt

class Dynamics:

    def __init__(self, dyn_freq=1000):
        # Set time step
        self.dt_ = 1/dyn_freq

    def SetState(self, state):
        self.state_ = state

    def SetInput(self, input):
        self.input_ = input

    def GetState(self):
        return self.state_

    def GetInput(self):
        return self.input_

    def __getitem__(self,key):
        return self.__getattribute__[key]

    def __setitem__(self, key, value):
        self.__setattr__[key] = value