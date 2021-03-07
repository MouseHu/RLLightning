import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Discrete

from agent.base_agent import Agent
from network.dqn_model import DQN


class DQNAgent(Agent, nn.Module):
    def __init__(self, args, component) -> None:
        Agent.__init__(self, component.env, component.eval_env, component.buffer, args)
        nn.Module.__init__(self)
        self.gamma = args.gamma
        obs_shape = self.env.observation_space.shape
        n_actions = self.env.action_space.n

        assert isinstance(self.env.action_space, Discrete), "DQN only support discrete action spaces!"

        self.net = DQN(obs_shape, n_actions)
        self.target_net = DQN(obs_shape, n_actions)
        self.update_target()

    def get_action(self, state, epsilon: float, train=True) -> int:
        if np.random.random() < epsilon:
            env = self.env if train else self.eval_env
            action = env.action_space.sample()
        else:
            q_values = self.policy(state)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())

        return action

    def policy(self, state):
        return self.net(state)

    def forward(self, state):
        return self.net(state)

    def parameters(self):
        return self.net.parameters()

    def compute_loss(self, batch, nb_batch, optimizer_idx=0) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        states, actions, rewards, dones, next_states = batch

        q_t = self.net(states).gather(1, actions).squeeze(-1)
        with torch.no_grad():
            q_tp1 = self.target_net(next_states).max(1)[0]
            q_tp1[dones.squeeze()] = 0.0
            q_tp1 = q_tp1.detach()

        q_target = q_tp1 * self.gamma + rewards

        # loss = nn.SmoothL1Loss()(q_t.float(), q_target.float())
        loss = nn.MSELoss()(q_t.float(), q_target.float())

        train_info = {
            "loss": loss,
            "q_mean": torch.mean(q_t),
            "q_target_mean": torch.mean(q_target)
        }
        return loss, train_info

    def update_target(self):
        self.target_net.load_state_dict(self.net.state_dict())
