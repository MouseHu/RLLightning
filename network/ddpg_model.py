import torch
import torch.nn as nn
import torch.nn.functional as F
from network.basic_model import MLP

# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971
# [Not the implementation used in the TD3 paper]

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, min_action=0):
        super(Actor, self).__init__()
        self.extractor = MLP([state_dim, 400, 300,action_dim])

        self.action_range = max_action - min_action
        self.min_action = min_action

    def forward(self, state):
        a = self.extractor(state)
        return self.action_range * torch.tanh(a)+self.min_action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.extractor = MLP([state_dim, 400, 300, 1])

    def forward(self, state, action):
        return self.extractor(state)
