import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Discrete
from typing import Dict

from agent.base_agent import Agent
from network.dqn_model import DQN, DuelingDQN, NoisyDQN


class C51Agent(Agent, nn.Module):
    """
    Implementation of Deep Q-learning. Includes dueling DQN, double DQN and noisy DQN
    """

    def __init__(self, args, component) -> None:
        Agent.__init__(self, component.env, component.eval_env, component.buffer, args)
        nn.Module.__init__(self)
        self.gamma = args.gamma
        obs_shape = self.env.observation_space.shape
        n_actions = self.env.action_space.n

        assert isinstance(self.env.action_space, Discrete), "DQN only support discrete action spaces!"

        network = {
            "dqn": DQN,
            "dueling": DuelingDQN,
            "noisy": NoisyDQN
        }.get(args.network, DQN)

        self.net = network(obs_shape, n_actions)
        self.target_net = network(obs_shape, n_actions)

        self.update_target()

    def get_action(self, state, epsilon: float, train=True) -> int:
        if "noisy" in self.args.network:
            # if there is already noise in network, ignore epsilon-greedy exploration
            epsilon = 0
        if np.random.random() < epsilon:
            env = self.env if train else self.eval_env
            action = env.action_space.sample()
        else:
            a = self.model(state) * self.supports
            a = a.sum(dim=2).max(1)[1].view(1, 1)
            return a.item()

        return action

    def policy(self, state):
        return self.net(state)

    def forward(self, state):
        return self.net(state)

    def parameters(self):
        return self.net.parameters()

    def compute_loss(self, batch, nb_batch, optimizer_idx=0) -> [torch.Tensor, Dict]:
        states, actions, rewards, dones, next_states, *rest = batch
        actions, dones, rewards = actions.long(), dones.long(), rewards.squeeze()

        current_dist = self.model(states).gather(1, actions).squeeze()

        target_prob = self.projection_distribution(batch)

        category_loss = -(target_prob * current_dist.log()).sum(-1)

        if "prioritized" in self.args.buffer:
            weight, idxes = rest
            self.component.buffer.update_priority((category_loss * weight).cpu().numpy())
            loss = torch.dot(category_loss, weight)
        else:
            loss = category_loss.mean()

        train_info = {
            "loss": loss,
        }
        return loss, train_info

    def update_target(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def projection_distribution(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, weights, indices = batch_vars

        with torch.no_grad():
            max_next_dist = torch.zeros((self.batch_size, 1, self.atoms), device=self.device,
                                        dtype=torch.float) + 1. / self.atoms
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                max_next_dist[non_final_mask] = self.target_net(non_final_next_states).gather(1, max_next_action)
                max_next_dist = max_next_dist.squeeze()

            Tz = batch_reward.view(-1, 1) + (self.gamma ** self.nsteps) * self.supports.view(1, -1) * non_final_mask.to(
                torch.float).view(-1, 1)
            Tz = Tz.clamp(self.v_min, self.v_max)
            b = (Tz - self.v_min) / self.delta
            l = b.floor().to(torch.int64)
            u = b.ceil().to(torch.int64)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            offset = torch.linspace(0, (self.batch_size - 1) * self.atoms, self.batch_size).unsqueeze(dim=1).expand(
                self.batch_size, self.atoms).to(batch_action)
            m = batch_state.new_zeros(self.batch_size, self.atoms)
            m.view(-1).index_add_(0, (l + offset).view(-1),
                                  (max_next_dist * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1),
                                  (max_next_dist * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        return m

    def get_max_next_state_action(self, next_states):
        next_dist = self.target_model(next_states) * self.supports
        return next_dist.sum(dim=2).max(1)[1].view(next_states.size(0), 1, 1).expand(-1, -1, self.atoms)
