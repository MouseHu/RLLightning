from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Box

from agent.base_agent import Agent


class ActorCriticAgent(Agent, nn.Module):
    """
    Parent Class of Actor-Critic Algorithms like DDPG,SAC and TD3
    """

    def __init__(self, args, component) -> None:
        Agent.__init__(self, component.env, component.eval_env, component.buffer, args)
        nn.Module.__init__(self)
        self.component = component
        self.gamma = args.gamma

        assert isinstance(self.env.action_space, Box), "Currently actor-critic only support continuous action spaces!"
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        self.actor = self.critic = None

    def policy(self, state):
        return self.actor(state)

    def forward(self, state):
        return self.actor(state), self.critic(state)

    def parameters(self):
        return chain(self.actor.parameters(), self.critic.parameters())

    def get_action(self, state, epsilon: float, train=True) -> int:
        if np.random.random() < epsilon:
            env = self.env if train else self.eval_env
            action = env.action_space.sample()
        else:
            action = self.actor(state).cpu().data.numpy().flatten()
            if train:
                action = (action + np.random.normal(0, self.max_action * self.args.explore_noise, size=self.action_dim)
                          ).clip(-self.max_action, self.max_action)
        return action

    def compute_loss(self, batch, optimizer_idx) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def update_target(self, tau=1):
        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    @torch.no_grad()
    def step(self, state, epsilon: float = 0.0, train=True):

        action = self.get_action(state, epsilon, train)

        # do step in the environment
        env = self.env if train else self.eval_env
        new_state, reward, done, info = env.step(action)
        if train:
            truly_done = info.get("truly_done", done)
            self.replay_buffer.add(self.state, action, reward, truly_done, new_state)

        if train:
            self.state = new_state
        else:
            self.eval_state = new_state

        if done:
            self.reset(train)
        return new_state, reward, done, info
