#!/usr/bin/env python

import numpy as np

class State:
    def __init__(self, dim_pos, dim_vel):
        self.name = self.__class__.__name__
        self.position = np.zeros((dim_pos,1))
        self.velocity = np.zeros((dim_vel,1))

    def __getitem__(self,key):
        return self.__getattribute__[key]

    def __setitem__(self, key, value):
        self.__setattr__[key] = value