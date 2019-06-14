#!/usr/bin/env python

# Import modules
import numpy as np

# Import modules
from state import *

class PID:
    # Model variables
    dt_ = 0.01   # time step for the integral part

    def __init__(self, state, output):
        self.name = self.__class__.__name__

        # Init current and desired state
        self.curr_state_ = state # curr_state might be either an internal variable (useful for the integrator) or not
        self.desired_state_ = self.curr_state_
        
        # Init current input
        self.SetOutput(output)
        
        # Init Matrices
        self.P_ = np.zeros((self.output_.size,self.curr_state_.position.size))
        # self.I_ = np.zeros((output,input)) # for the integrative I've to think a bit more about to define it
        self.D_ = np.zeros((self.output_.size,self.curr_state_.velocity.size))

    def SetGains(self, P, D):
        if P.shape == self.P_.shape:
            self.P_ = P
        else:
            raise ValueError('Wrong matrix dimension')

        # if I.shape == self.I_.shape:
        #     self.I_ = I
        # else:
        #     raise ValueError('Wrong matrix dimension')

        if D.shape == self.D_.shape:
            self.D_ = D
        else:
            raise ValueError('Wrong matrix dimension')

    def SetDesired(self, state):
        self.desired_state_ = state

    def GetDesired(self):
        return self.desired_state_

    def SetOutput(self, output):
        self.output_ = output

    def GetOutput(self):
        return self.output_

    def Update(self, state):
        self.output_ = -self.P_*(state.position - self.desired_state_.position) - -self.D_*(state.velocity - self.desired_state_.velocity)

    