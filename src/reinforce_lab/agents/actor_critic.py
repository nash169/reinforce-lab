#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


class ActorCritic(nn.Module):
    alpha_ = 0.1

    def __init__(
        self, num_inputs, num_outputs, hidden_size=10, std=0.5
    ):  # def __init__(self, num_inputs, num_outputs, hidden_size, std=0.5):
        super(ActorCritic, self).__init__()

        self.outputs_ = num_outputs

        self.critic_ = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

        self.actor_ = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2 * num_outputs),
        )

        self.critic_.apply(self.init_weights)
        self.actor_.apply(self.init_weights)

    def forward(self, x):
        value = self.critic_(x)
        actor = self.actor_(x)

        # self.output_ = actor
        # print(actor)

        dist = Normal(actor[:, 0 : self.outputs_], actor[:, self.outputs_ :].exp())

        return dist, value

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.1)
            nn.init.constant_(m.bias, 0.1)
