#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size=10, std=0.5): # def __init__(self, num_inputs, num_outputs, hidden_size, std=0.5):
        self.num_inputs_ = num_inputs
        self.num_outputs_ = num_outputs
        
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Tanh()
        )

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        # self.apply(init_weights)
        
    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)

        std   = self.log_std.exp().expand_as(mu)

        # actor_out[:,0:self.num_outputs_] = torch.FloatTensor(np.tanh(actor_out[:,0:self.num_outputs_].detach().numpy()))
        # dist  = Normal(actor_out[:,0:self.num_outputs_], actor_out[:,(self.num_outputs_+1):])

        dist  = Normal(mu, std)

        return dist, value