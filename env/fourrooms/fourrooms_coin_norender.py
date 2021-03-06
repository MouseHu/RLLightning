"""Fourrooms Game with Coins 


class:
    + FourroomsCoinState and FourroomsCoin are based on FourroomsBase,
    you can inherit them for extension.
    + FourroomsCoinNorender support rendering.
    + FourroomsCoinBackgroundNoise is an extension example.

------
Design principles:
- seperate state,observation and rendering
- Each heirarchy should be a complete game.
- easy saving and loading
- gym interface

------
Some possible extensions:
- To change render colors or add background noises,rewrite render functions.
NOTE:ANY change of layout size should accompany a redefination of observation_space or obs_height and obs width.

- To add enemys,inherit FourroomsCoin.
- To change game layout,rewrite init_layout.
- This file includes an extension example FourroomsCoinBackgroundNoise.

------
Test scripts:
- check_render(env):check rendering
- check_env(env):imported from stable_baselines,check whether it follows gym interface
- check_run(env):random run
- I also try to train an agent with stable_baselines to check the difficulty and reasonability of game.
"""

import gym
import time
from gym import error, core,spaces
from gym.envs.registration import register
import random
import numpy as np
from env.fourrooms import *
from env.wrappers import ImageInputWarpper
from copy import copy,deepcopy
import abc
import cv2
import time

class FourroomsCoinState(FourroomsBaseState):
    """
    The class that contains all information needed for restoring a FourroomsCoin game.
    The saving and restoring game must be of the same class and parameters.
    ···
    Attributes:
    position_n: int
        The numeralized position of agent.
    current_step: int
    goal_n:int
        The numeralized position of goal.
    done: bool
    coin_dict: dict int->(int,bool)
        coin->(value,if_exist)
    ...
    
    """
    def __init__(self,position_n:int,current_steps:int,goal_n:int,done:bool,num_pos:int,\
    coin_dict:dict,num_coins,cum_reward:list):
        self.position_n=position_n
        self.current_steps=current_steps
        self.goal_n=goal_n
        self.done=done
        self.num_pos=num_pos
        self.coin_dict=coin_dict
        self.num_coins=num_coins
        self.cum_reward=cum_reward
        
    def __init__(self,base:FourroomsBase,coin_dict,num_coins,cum_reward):
        self.position_n=base.position_n
        self.current_steps=base.current_steps
        self.goal_n=base.goal_n
        self.done=base.done
        self.num_pos=base.num_pos
        self.coin_dict=coin_dict
        self.num_coins=num_coins
        self.cum_reward=cum_reward

    def coined_state(self):
        num_coins=0
        for k,v in self.coin_dict.items():
            num_coins+=1
        value_list=[(v[0] if v[1] else 0) for v in self.coin_dict.values()]
        multiplier = np.dot(value_list, [2 ** i for i in range(num_coins)])
        return multiplier * self.num_pos + self.position_n

    def to_obs(self)->np.array:
        return np.array(self.coined_state())

class FourroomsCoin(FourroomsNorender):
    """Fourroom game with agent,goal and coins that inherits gym.Env.

    This class should not render.
    """
    def __init__(self, max_epilen=100, goal=None, num_coins=3,seed=0):
        #这里为了兼容留下了random coin，实际上没有用
        
        super(FourroomsCoin, self).__init__(max_epilen, goal,seed=seed)
        self.num_coins=num_coins
        assert self.num_pos > (num_coins + 5),"too many coins"
        self.observation_space = spaces.Discrete(self.num_pos * (2 ** num_coins))
        coin_list = np.random.choice(self.init_states, num_coins, replace=False)
        #You can change the value of coin here
        coin_dict = {coin:(1,True) for coin in coin_list}
        # random encode
        super().reset()
        self.state=FourroomsCoinState(self.state,coin_dict,num_coins,cum_reward=[])
        
        
    def step(self, action):
        if not self.open:
            raise Exception("Environment should be reseted")
        try:
            nextcell = tuple(self.currentcell + self.directions[action])
        except TypeError:
            nextcell = tuple(self.currentcell + self.directions[action[0]])

        if not self.occupancy[nextcell]:
            self.currentcell = nextcell

        state = self.tostate[self.currentcell]
        self.state.position_n=state
        if state == self.state.goal_n:
            reward=10
        elif self.state.coin_dict.get(state,(0,False))[1]:#if find coin
            reward=self.state.coin_dict.get(state, 0)[0]*10
            self.state.coin_dict[state] = (self.state.coin_dict[state][0],False)
        else:
            reward = -0.1
        self.state.cum_reward.append(reward)            

        self.state.current_steps += 1
        self.state.done = (state == self.state.goal_n) or self.state.current_steps >= self.max_epilen

        info = {}
        if self.state.done:
            info = {'episode': {'r': np.sum(self.state.cum_reward), 'l': self.state.current_steps}}
            self.state.cum_reward = []
            self.open=False

        return self.state.to_obs(), reward, self.state.done, info

    def reset(self):
        super().reset()
        self.state=FourroomsCoinState(self.state,{},self.num_coins,[])
        init_states=deepcopy(self.init_states)
        if self.state.goal_n in init_states:
            init_states.remove(self.state.goal_n)
        if self.state.position_n in init_states:
            init_states.remove(self.state.position_n)
        coin_list = np.random.choice(init_states, self.num_coins, replace=False)
        coin_dict = {coin: (1,True) for coin in coin_list}
        
        self.state.coin_dict=coin_dict

        return self.state.to_obs()
    @abc.abstractmethod
    def render(self):
        pass

class FourroomsCoinNorender(FourroomsCoin):
    def __init__(self, max_epilen=100, goal=None, num_coins=3,seed=0):
        super().__init__(max_epilen=max_epilen, goal=goal, num_coins=num_coins,seed=seed)

    def render(self, mode=0):
        blocks = []
        for coin, count in self.state.coin_dict.items():
            x, y = self.tocell[coin]
            if count[1]:#exist
                blocks.append(self.make_block(x, y, (0, 1, 0)))
        blocks.extend(self.make_basic_blocks())
        
        arr = self.render_with_blocks(self.origin_background, blocks)

        return arr

# class FourroomsCoinNorender(FourroomsNorender):
#     def __init__(self, max_epilen=400,obs_size=128,seed=10):
#         super(FourroomsCoinNorender, self).__init__(max_epilen)
#         #self.observation_space = spaces.Discrete(self.num_pos * 2)
#         self.obs_size = obs_size
#         self.obs_height = obs_size
#         self.obs_width = obs_size
# 	    #random coin
#         self.coin = np.random.choice(self.init_states)
#         self.have_coin = True
#         self.init_states.remove(self.coin)
#         self.mapping = np.arange(self.num_pos * 2)
#         self.dict = np.zeros((self.observation_space.n, 3))
#         self.state_space_capacity = self.observation_space.n
#         self.get_dict()
#         self.max_steps=max_epilen
#         self.previous_action=0
#         self.seed(seed)

#     def step(self, action):
        
#         try:
#             nextcell = tuple(self.currentcell + self.directions[action])
#         except TypeError:
#             nextcell = tuple(self.currentcell + self.directions[action[0]])

#         if not self.occupancy[nextcell]:#if not wall
#             self.currentcell = nextcell
#             if np.random.uniform() < 1/3.:#not deterministic
#                 empty_cells = self.empty_around(self.currentcell)
#                 self.currentcell = empty_cells[np.random.randint(len(empty_cells))]

#         state = self.tostate[self.currentcell]
#         reward = 1. if (state % self.num_pos == self.coin and self.have_coin) or state % self.num_pos == self.goal else 0.
#         self.state=state
#         if state == self.coin:
#             self.have_coin = False
#         if not self.have_coin:
#             state += self.num_pos
#         self.current_steps += 1

#         self.done = (state % self.num_pos == self.goal)#until find goal
#         if self.current_steps >= self.max_epilen and self.done==False:
#             self.done = True

#         info = {}
#         if self.done:#for plotting
#             info = {'episode': {'r': (state % self.num_pos == self.goal)+1-self.have_coin, 'l': self.current_steps}}
#         return np.array(self.mapping[state]), reward, self.done, info

#     def reset(self, state=-1):
#         if state < 0:
#             state = np.random.choice(self.init_states)
#         if state >= self.num_pos or state == self.coin:
#             self.have_coin = False
#         else:
#             self.have_coin = True
#         self.state=state
#         self.currentcell = self.tocell[state % self.num_pos]
#         self.done = False
#         self.current_steps = 0
#         return np.array(self.mapping[state])

#     def render(self, state=-1):
#         arr=self.render_origin()
#         #expand to dim,a tmp workaround
#         obs= np.zeros((self.obs_size, self.obs_size, 3),dtype=np.int)
#         padding_height,padding_width = (obs.shape[0]-arr.shape[0])//2,(obs.shape[1]-arr.shape[1])//2
#         obs[padding_height:padding_height+arr.shape[0],padding_width:padding_width+arr.shape[1],:] = arr
#         #obs.shape:(128,128,3)
#         return obs

#     def render_origin(self,blocks=[]):
#         #WARNING:blocks must be exlicitly passed
#         if self.have_coin:
#             x, y = self.tocell[self.coin]
#             blocks.append(self.make_block(x, y, (0, 1, 0)))
        
#         arr = super().render(blocks=blocks)
#         #arr.shape:(104,104,3)
#         return arr

# an extension example
class FourroomsCoinBackgroundNoise(FourroomsCoinNorender):
    def __init__(self, max_epilen=400, obs_size=128,seed=0):
        super(FourroomsCoinBackgroundNoise, self).__init__(max_epilen,seed=seed)
        self.obs_height=obs_size
        self.obs_width=obs_size
        self.background = np.zeros((2, obs_size, obs_size, 3),dtype=np.int)
        self.background[0, :, :, 1] = 127  # red background
        self.background[1, :, :, 2] = 127  # blue background

    def render(self,mode=0):
        which_background = self.state.position_n % 2
        obs = deepcopy(self.background[which_background, ...])
        arr = super().render()
        padding_height,padding_width = (obs.shape[0]-arr.shape[0])//2,(obs.shape[1]-arr.shape[1])//2
        obs[padding_height:padding_height+arr.shape[0],padding_width:padding_width+arr.shape[1],:] = arr
        return obs

if __name__=='__main__':

    #print(FourroomsCoinNorender.__mro__)
    env=ImageInputWarpper(FourroomsCoinNorender(seed=time.time()))
    check_render(env)
    check_env(env,warn=True)
    check_run(env)
    print("basic check finished")

    #stable-baseline test
    env = make_vec_env(lambda: env, n_envs=1)
    model = ACKTR('CnnPolicy',env, verbose=1)
    model.learn(total_timesteps=300000)
    
    env=ImageInputWarpper(FourroomsCoinNorender(seed=time.time()))
    obs = env.reset()
    reward_list=[]
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        reward_list.append(rewards)
        if dones:
            obs=env.reset()
            print("i={},done\n".format(i))
            print("reward is: "+str(np.sum(reward_list))+'\n')
            reward_list=[]