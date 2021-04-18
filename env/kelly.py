import numpy as np
from gym import Env
from gym.spaces import Box


class Kelly(Env):
    def __init__(self, win_rate, odds, epi_length):
        super().__init__()
        assert 0 < win_rate < 1
        assert odds > 0
        self.win_rate = win_rate
        self.odds = odds
        self.epi_length = epi_length
        self.num_steps = 0
        self.log_reward = 0
        self.observation_space = Box(low=(-np.inf, 0.,), high=(np.inf, epi_length), dtype=np.float32)
        self.action_space = Box(low=(0.,), high=(1.,), dtype=np.float32)

    def reset(self):
        self.num_steps, self.log_reward = 0, 0
        return np.array([self.num_steps, self.log_reward])

    def step(self, action: float):
        bet = np.random.rand()
        if bet <= self.win_rate:
            # wins
            self.log_reward += np.log(1 + self.odds * action)
        else:
            # loses
            self.log_reward -= np.log(1 - action)
        self.epi_length += 1
        return np.array([self.num_steps, self.log_reward])

    def render(self, mode='human'):
        pass

