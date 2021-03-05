import numpy as np
import torch.nn as nn
from network.basic import *


class DQN(nn.Module):
    def __init__(self, obs_shape: tuple, n_actions: int, hidden_size: tuple = (64,), features_dim: int = 512):
        """
        Args:
            obs_shape: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super(DQN, self).__init__()
        # print(obs_shape)
        if len(obs_shape) == 1:
            obs_size = obs_shape[0]
            self.net = MLP((obs_size,) + hidden_size + (n_actions,))
        else:
            self.net = nn.Sequential(
                NatureCNN(obs_shape, features_dim),
                MLP((features_dim,) + hidden_size + (n_actions,))
            )

    def forward(self, x):
        return self.net(x.float())
