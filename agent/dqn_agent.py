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

    def policy(self, state):
        return self.net(state)

    def forward(self, state):
        return self.net(state)

    def parameters(self):
        return self.net.parameters()

    def compute_loss(self, batch) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        states, actions, rewards, dones, next_states = batch
        # print(states.shape,dones.shape,next_states.shape)
        q_t = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        # print(next_states.shape)
        with torch.no_grad():
            q_tp1 = self.target_net(next_states).max(1)[0]
            q_tp1[dones] = 0.0
            q_tp1 = q_tp1.detach()

        q_target = q_tp1 * self.gamma + rewards

        # return nn.SmoothL1Loss()(q_t.float(), q_target.float()), torch.mean(q_t), torch.mean(q_target)
        return nn.MSELoss()(q_t.float(), q_target.float()), torch.mean(q_t), torch.mean(q_target)

    def update_target(self):
        self.target_net.load_state_dict(self.net.state_dict())
