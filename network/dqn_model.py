from network.basic_model import *


class DQN(nn.Module):
    def __init__(self, obs_shape: tuple, n_actions: int, hidden_size: tuple = (64,), features_dim: int = 512):
        super(DQN, self).__init__()
        if len(obs_shape) == 1:
            # vector input, use MLP
            obs_size = obs_shape[0]
            self.net = MLP((obs_size,) + hidden_size + (n_actions,))
        else:
            # image input, use CNN
            self.net = nn.Sequential(
                NatureCNN(obs_shape, features_dim),
                nn.Linear(features_dim, n_actions)
            )

    def forward(self, x):
        return self.net(x.float())
