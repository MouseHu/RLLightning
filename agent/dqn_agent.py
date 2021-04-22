import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Discrete
from typing import Dict

from agent.base_agent import Agent
from network.dqn_model import DQN, DuelingDQN, NoisyDQN


class DQNAgent(Agent, nn.Module):
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

    def compute_loss(self, batch, nb_batch, optimizer_idx=0) -> [torch.Tensor, Dict]:
        states, actions, rewards, dones, next_states, *rest = batch
        actions, dones, rewards = actions.long(), dones.long(), rewards.squeeze()

        q_t = self.net(states).gather(1, actions).squeeze(-1)
        with torch.no_grad():
            if self.args.double:
                action_tp1 = self.net(next_states).max(1)[1].unsqueeze(-1)
                q_tp1 = self.target_net(next_states)
                v_tp1 = q_tp1.gather(1, action_tp1).squeeze(-1)
            else:
                v_tp1 = self.target_net(next_states).max(1)[0]

            v_tp1[dones.squeeze()] = 0.0
            v_tp1 = v_tp1.detach()

        q_target = v_tp1 * self.gamma + rewards

        # loss = nn.SmoothL1Loss()(q_t.float(), q_target.float())
        td_loss = (q_t.float() - q_target.float()) ** 2

        if "prioritized" in self.args.buffer:
            weight, idxes = rest
            self.component.buffer.update_priority((td_loss * weight).cpu().numpy())
            loss = torch.dot(td_loss, weight)
        else:
            loss = td_loss.mean()
        train_info = {
            "loss": loss,
            "q_mean": torch.mean(q_t),
            "q_target_mean": torch.mean(q_target)
        }
        return loss, train_info

    def update_target(self):
        self.target_net.load_state_dict(self.net.state_dict())

    @torch.no_grad()
    def step(self, epsilon: float = 0.0, train=True):
        state = self.get_state(train = train)
        action = self.get_action(state, epsilon, train)
        # do step in the environment
        env = self.env if train else self.eval_env
        new_state, reward, done, info = env.step(action)
        #FIXME:这一步似乎应该交给leaner操作
        if train:
            self.replay_buffer.add(self.state, action, reward, done, new_state)

        if train:
            self.state = new_state
        else:
            self.eval_state = new_state

        if done:
            self.reset(train)
        return new_state, reward, done, info
