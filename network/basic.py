import numpy as np
import torch.nn as nn
import torch as th


class NatureCNN(nn.Module):
    def __init__(self, obs_shape, features_dim=512):
        super(NatureCNN, self).__init__()
        self.obs_shape = obs_shape
        self.features_dim = features_dim

        n_input_channels = obs_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(np.zeros(obs_shape)).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class MLP(nn.Module):
    def __init__(self, layers, layer_norm=False):
        super().__init__()
        self.layers = layers
        self.layer_norm = layer_norm
        self.mlps = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(self.layers) - 1)])

    def forward(self, x):
        for i, linear in enumerate(self.mlps):
            x = linear(x)
            if i < len(self.mlps) - 1:
                x = nn.ReLU()(x)
                # if self.layer_norm:
                #     x = nn.LayerNorm(x)

        return x
