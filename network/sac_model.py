import numpy as np
import torch
import torch.nn as nn

from network.basic_model import MLP

EPS = 1e-6  # Avoid NaN (prevents division by zero or log of zero)


def gaussian_entropy(log_std):
    return (log_std + 0.5 * np.log(2.0 * np.pi * np.e)).sum(dim=-1)


def gaussian_likelihood(pi, mu, log_std):
    pre_sum = -0.5 * (((pi - mu) / (log_std.exp() + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return pre_sum.sum(dim=1)


def apply_squashing_func(mu, pi, logp_pi):
    # Squash the output
    deterministic_policy = torch.tanh(mu)
    policy = torch.tanh(pi)
    # OpenAI Variation:
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    # logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - policy ** 2, lower=0, upper=1) + EPS), axis=1)
    # Squash correction (from original implementation)
    logp_pi -= (1 - policy ** 2 + EPS).log().sum(dim=1)
    return deterministic_policy, policy, logp_pi


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, log_std_bounds=(-20, 2)):
        super(Actor, self).__init__()

        self.hidden = MLP([state_dim, 256, 256])
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

        self.log_std_bounds = log_std_bounds
        self.max_action = max_action

    def forward(self, state):
        hidden = self.hidden(state)
        mu = self.mu(hidden)
        log_std = self.log_std(hidden)
        std = torch.clip(log_std, self.log_std_bounds[0], self.log_std_bounds[1]).exp()
        pi = mu + torch.normal(0, 1, mu.shape).type_as(mu) * std
        logp_pi = gaussian_likelihood(pi, mu, log_std)
        entropy = gaussian_entropy(log_std)
        deterministic_policy, policy, logp_pi = apply_squashing_func(mu, pi, logp_pi)

        return deterministic_policy, policy, logp_pi, entropy


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.q1 = MLP([state_dim + action_dim, 256, 256, 1])
        # Q2 architecture
        self.q2 = MLP([state_dim + action_dim, 256, 256, 1])
        # Value architecture
        self.value = MLP([state_dim, 256, 1])

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1,q2,v = self.q1(sa),self.q2(sa),self.value(state)
        return q1, q2, v

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa)

    def Value(self,state):
        return self.value(state)
