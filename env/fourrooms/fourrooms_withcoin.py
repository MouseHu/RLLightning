import numpy as np
import gym
import time
from gym import error, spaces
from gym import core, spaces
from gym.envs.registration import register
import random
#from attn_toy.env.rendering import *
from attn_toy.env.fourrooms import Fourrooms,FourroomsNorender
from copy import copy,deepcopy
import cv2

class FourroomsCoin(Fourrooms):
    """Fourroom game with agent,goal and coin.

    ···
    Attributes:
    ------------
    
    """
    def __init__(self, max_epilen=400):
        super(FourroomsCoin, self).__init__(max_epilen)
        self.observation_space = spaces.Discrete(self.num_pos * 2)
        self.coin = 15
        self.have_coin = True
        self.init_states.remove(self.coin)
        # random encode
        self.mapping = np.arange(self.num_pos * 2)
        # self.mapping = self.mapping % self.num_pos
        self.dict = np.zeros((self.observation_space.n, 3))
        self.state_space_capacity = self.observation_space.n
        self.get_dict()

    def step(self, action):

        try:
            nextcell = tuple(self.currentcell + self.directions[action])
        except TypeError:
            nextcell = tuple(self.currentcell + self.directions[action[0]])

        if not self.occupancy[nextcell]:
            self.currentcell = nextcell
            if np.random.uniform() < 0:
                # if self.rng.uniform() < 1/3.:
                empty_cells = self.empty_around(self.currentcell)
         
                self.currentcell = empty_cells[np.random.randint(len(empty_cells))]

        state = self.tostate[self.currentcell]
        reward = 1. if (state % self.num_pos == self.coin and self.have_coin) or state % self.num_pos == self.goal else 0.
        if state == self.coin:
            self.have_coin = False
        if not self.have_coin:
            state += self.num_pos
        self.current_steps += 1
        if self.current_steps >= self.max_epilen:
            self.done = True
        self.done = (state % self.num_pos == self.goal)

        info = {}
        if self.done:
            # print(self.current_step, state == self.goal, self.max_epilen)
            info = [{'episode': {'r': state == self.goal, 'l': self.current_steps}}]
        # print(self.currentcell)
        return np.array(self.mapping[state]), reward, self.done, info

    def reset(self, state=-1):
        # state = self.rng.choice(self.init_states)
        # self.viewer.close()
        if state < 0:
            state = np.random.choice(self.init_states)
        if state >= self.num_pos or state == self.coin:
            self.have_coin = False
        else:
            self.have_coin = True
        self.currentcell = self.tocell[state % self.num_pos]
        self.done = False
        self.current_steps = 0
        return np.array(self.mapping[state])

    def render(self, mode=0):

        if self.have_coin:
            x, y = self.tocell[self.coin]
            self.add_block(x, y, (0, 1, 0))
        if self.currentcell[0] > 0:
            x, y = self.currentcell
            self.add_block(x, y, (0, 0, 1))

        x, y = self.tocell[self.goal]
        self.add_block(x, y, (1, 0, 0))
        # self.viewer.
        arr = self.viewer.render(return_rgb_array=True)

        return arr


class FourroomsCoinDynamicNoise(FourroomsCoin):
    def __init__(self, max_epilen=100, obs_size=128):
        super(FourroomsCoinDynamicNoise, self).__init__(max_epilen)
        self.background = np.zeros((2, obs_size, obs_size, 3),dtype=np.int)
        # self.background[0, :, :, 1] = 127  # red background
        # self.background[0, :, :, 2] = 127  # red background
        # self.background[1, :, :, 1] = 127  # blue background
        # self.background[1, :, :, 2] = 127  # blue background

    def render(self, state=-1):
        which_background = state % 2
        # print(state,which_background)
        obs = copy(self.background[which_background, ...])
        arr = super(FourroomsCoinDynamicNoise,self).render(state)
        padding_height,padding_width = (obs.shape[0]-arr.shape[0])//2,(obs.shape[1]-arr.shape[1])//2
        obs[padding_height:padding_height+arr.shape[0],padding_width:padding_width+arr.shape[1],:] = arr
        return obs
        
if __name__=='__main__':
    pass

