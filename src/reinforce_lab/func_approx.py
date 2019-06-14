#!/usr/bin/env python

# Import modules
import numpy as np
import tensorflow as tf

# Import classes

class FuncApprox:

    def __init__(self, input_dim, output_dim):
        self.input_dim_ = input_dim
        self.output_dim_ = output_dim

    def GetInput(self):
        return self.input_

    def GetOutput(self):
        return self.output_

    def SetInput(self, input_to_set):
        self.input_ = input_to_set

    def SetOutput(self, output_to_set):
        self.output_ = output_to_set



            