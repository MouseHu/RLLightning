import gym
import numpy as np
import torch

from buffer.replay_buffer import ReplayBuffer


class Agent(object):
    def __init__(self, env: gym.Env, eval_env: gym.Env, replay_buffer: ReplayBuffer, args) -> None:
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.env = env
        self.eval_env = eval_env
        self.replay_buffer = replay_buffer
        self.args = args
        self.state = None
        self.eval_state = None
        self.reset()

    def policy(self, state):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

    def reset(self, train=True) -> None:
        """Resets the environment and updates the state"""
        if train:
            self.state = self.env.reset()
        else:
            if self.eval_env is not None:
                self.eval_state = self.eval_env.reset()

    def get_action(self, state, epsilon: float, train=True) -> int:
        if np.random.random() < epsilon:
            env = self.env if train else self.eval_env
            action = env.action_space.sample()
        else:
            q_values = self.policy(state)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())

        return action

    @torch.no_grad()
    def step(self, state, epsilon: float = 0.0, train=True):

        action = self.get_action(state, epsilon, train)

        # do step in the environment
        env = self.env if train else self.eval_env
        new_state, reward, done, info = env.step(action)
        if train:
            self.replay_buffer.add(self.state, action, reward, done, new_state)

        if train:
            self.state = new_state
        else:
            self.eval_state = new_state

        if done:
            self.reset(train)
        return new_state, reward, done, info
