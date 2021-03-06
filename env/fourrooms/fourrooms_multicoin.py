import numpy as np
import gym
import time
from gym import error, spaces
from gym import core, spaces
from gym.envs.registration import register
import random
# from attn_toy.env.rendering import *
from attn_toy.env.fourrooms import Fourrooms, FourroomsNorender
from copy import copy


class FourroomsMultiCoin(FourroomsNorender):
    def __init__(self, max_epilen=100, goal=77, num_coins=3, random_coin=False):
        super(FourroomsMultiCoin, self).__init__(max_epilen, goal)
        assert self.num_pos > (num_coins + 2)
        self.num_coins = num_coins
        self.observation_space = spaces.Discrete(self.num_pos * (2 ** num_coins))
        self.coin_list = np.random.choice(self.init_states, num_coins, replace=False)
        self.coin_dict = {coin: 1 for coin in self.coin_list}
        # random encode
        self.random_coin = random_coin
        self.mapping = np.arange(self.num_pos * 2)
        self.cum_reward = []
        # self.mapping = self.mapping % self.num_pos
        self.dict = np.zeros((self.observation_space.n, 3))
        self.get_dict()

    def coined_state(self, state):
        state = state % self.num_pos
        multiplier = np.dot(list(self.coin_dict.values()), [2 ** i for i in range(self.num_coins)])
        return multiplier * self.num_pos + state

    def step(self, action):

        try:
            nextcell = tuple(self.currentcell + self.directions[action])
        except TypeError:
            nextcell = tuple(self.currentcell + self.directions[action[0]])

        if not self.occupancy[nextcell]:
            self.currentcell = nextcell

        state = self.tostate[self.currentcell]

        if state == self.goal or self.coin_dict.get(state, 0) > 0:
            reward = 100.
        else:
            reward = -1.
        self.cum_reward.append(reward)
        if self.coin_dict.get(state, 0) > 0:
            self.coin_dict[state] -= 1

        self.num_steps += 1
        self.done = (state == self.goal) or self.num_steps >= self.max_epilen

        info = {}
        if self.done:
            # print(self.current_step, state == self.goal, self.max_epilen)
            info = {'episode': {'r': np.sum(self.cum_reward), 'l': self.current_steps}}
            self.cum_reward = []
        # print(self.currentcell)
        return self.coined_state(state), reward, self.done, info

    def reset(self, state=-1):
        # state = self.rng.choice(self.init_states)
        # self.viewer.close()
        self.done = False
        self.num_steps = 0
        if self.random_coin:
            self.coin_list = np.random.choice(self.init_states, self.num_coins, replace=False)
        if state > 0:
            self.coin_dict = {coin: int((state // self.num_pos) % (2 ** (i + 1)) == 1) for i, coin in
                              enumerate(self.coin_list)}
        else:
            self.coin_dict = {coin: 1 for coin in self.coin_list}
        if state < 0:
            state = np.random.choice(self.init_states)

        self.currentcell = self.tocell[state % self.num_pos]

        return self.coined_state(state)

    def render(self, mode=0):
        blocks = []
        for coin, count in self.coin_dict.items():
            x, y = self.tocell[coin]
            if count > 0:
                blocks.append(self.make_block(x, y, (0, 1, 0)))

        if self.currentcell[0] > 0:
            x, y = self.currentcell
            blocks.append(self.make_block(x, y, (0, 0, 1)))

        x, y = self.tocell[self.goal]
        blocks.append(self.make_block(x, y, (1, 0, 0)))
        # self.viewer.
        arr = self.render_with_blocks(self.origin_background, blocks)

        return arr


class FourroomsMultiCoinRandomNoise(FourroomsMultiCoin):  # noise type = optimal action
    def __init__(self, max_epilen=100, obs_size=128, seed=0, num_colors=200, goal=77):
        np.random.seed(seed)
        super(FourroomsMultiCoinRandomNoise, self).__init__(max_epilen, goal=goal)
        self.obs_size = obs_size
        self.obs_height = obs_size
        self.obs_width = obs_size

        self.num_colors = num_colors
        self.seed = seed
        self.color = np.random.randint(0, 255, (self.num_colors, 3))
        # self.observation_space = spaces.Discrete(self.num_pos)
        self.state_space_capacity = self.observation_space.n

    def render(self, state=-1):
        obs = np.tile(self.color[np.random.randint(0, self.num_colors)][np.newaxis, np.newaxis, :],
                      (self.obs_size, self.obs_size, 1))

        arr = super(FourroomsMultiCoinRandomNoise, self).render(state % self.num_pos)
        padding_height, padding_width = (obs.shape[0] - arr.shape[0]) // 2, (obs.shape[1] - arr.shape[1]) // 2
        obs[padding_height:padding_height + arr.shape[0], padding_width:padding_width + arr.shape[1], :] = arr
        return obs.astype(np.uint8)
