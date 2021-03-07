"""Basic Fourrooms Game 

This script contains a basic version of Fourrooms.

If you want to extend the game,please inherit FourroomsBaseState and FourroomsBase.

Fourrooms and FourroomsNorender support rendering in different ways.

Some design principles,extension advice and test information can be seen in fourrooms_coin_norender.py.

BUG:Loading and saving is not well-tested.
"""
import numpy as np
import gym
import time
from gym import error, spaces
from gym import core, spaces
from gym.envs.registration import register
import random
#from attn_toy.env.rendering import *
from copy import deepcopy
import abc
from env.wrapper import ImageInputWarpper
import cv2
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import ACER,A2C,ACKTR
from stable_baselines.common import make_vec_env
from stable_baselines.common.evaluation import evaluate_policy

class FourroomsBaseState(object):
    """State of FourroomsBase

    The class that contains all information needed for restoring a game.
    The saving and restoring game must be of the same class.
    This class is designed for FourroomsBase.
    ···
    Attributes:
    position_n: int
        The numeralized position of agent.
    current_step: int
    goal_n:int
        The numeralized position of goal.
    done: bool
    """
    def __init__(self,position_n:int,current_steps:int,goal_n:int,done:bool,num_pos:int):
        self.position_n=position_n
        self.current_steps=current_steps
        self.goal_n=goal_n
        self.done=done
        self.num_pos=num_pos

    def to_obs(self)->np.array:
        return np.array(self.position_n)

class FourroomsBase(gym.Env):
    metadata = {'render.modes': ['rgb_array']}
    """Fourroom game with agent and goal that inherits gym.Env.

    This class does not render.
    ···
    Attributes:
    ------------
    occupancy: map
        from (x,y) to [0,1],check whether the position is blocked.
    num_pos: int 
        the number of non-blocked positions
    block_size: int
        length of a squared block measured in pixels 
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Discrete
    tocell: dict (x,y)->state
        a map from position to state
    tostate: dict state->(x,y)
    init_states: list
        all non-blocked states to place objects,REMAIN CONSTANT
    max_epilen: int
        maximum episode length
    current_steps: int
        current step
    currentcell:(x,y)
        current cell
    unwrapped: self
        origin env
    state: FourroomsBaseState
        internal state
    observation: np.array
        part of the state
    open: bool
        whether game is running
    
    Methods: As gym interface
    ---------
    step
    reset
    close
    seed
    ...
    """
    def __init__(self, max_epilen=100, goal=None,seed=0):
        """
        goal:None means random goal
        """
        self.seed(seed)
        self.init_layout()
        self.init_basic(max_epilen,goal)
    def init_layout(self):
        self.layout = """\
1111111111111
1     1     1
1     1     1
1           1
1     1     1
1     1     1
11 1111     1
1     111 111
1     1     1
1     1     1
1           1
1     1     1
1111111111111
"""
        self.block_size = 3
        self.occupancy = np.array(
            [np.array(list(map(lambda c: 1 if c == '1' else 0, line))) for line in self.layout.splitlines()])
        self.num_pos = int(np.sum(self.occupancy == 0))
        self.Row = np.shape(self.occupancy)[0]
        self.Col = np.shape(self.occupancy)[1]
        self.obs_height = self.block_size * self.Row
        self.obs_width = self.block_size * self.Col

    def init_basic(self, max_epilen,goal):

        # From any state the agent can perform one of four actions, up, down, left or right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.num_pos)

        self.directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1)), np.array((0, 1))]

        self.rand_color = np.random.randint(0, 255, (200, 3))#low,high,size
        self.tostate = {}
        
        statenum = 0
        #label states
        for i in range(len(self.occupancy)):
            for j in range(len(self.occupancy[0])):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i, j)] = statenum
                    statenum += 1
        self.tocell = {v: k for k, v in self.tostate.items()}
        self.init_states = list(range(self.observation_space.n))
        self.dict =dict()

        self.max_epilen = max_epilen
        self.get_dict()
        
        self.reward_range = (0, 1)
        self.metadata = None
        self.allow_early_resets = True
        if goal!=None and goal>(self.observation_space.n):
            raise ValueError("invalid goal position")
        self.goal=goal
        self.open=False

    def get_dict(self):
        """
        Label positions of states
        TODO:add infomation(e.g. goal/agent) to positions
        """
        count = 0
        for i in range(self.Row):
            for j in range(self.Col):
                if self.occupancy[i, j] == 0:
                    # code
                    self.dict[count] = (i,j)
                    count += 1

    def empty_around(self, cell:tuple)->list:
        """
        Find all available cells around the cell.
        """
        avail = []
        for action in range(self.action_space.n):
            nextcell = tuple(cell + self.directions[action])
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
        return avail

    def reset(self):
        """
        reset state,rechoose goal position if needed
        """
        self.open=True

        init_states=deepcopy(self.init_states)
        if self.goal==None:
            goal=np.random.choice(init_states)
        else:
            goal=self.goal
        init_states.remove(goal)
        init_position = np.random.choice(init_states)
        self.currentcell = self.tocell[init_position]
        self.state = FourroomsBaseState(position_n=init_position,current_steps=0,goal_n=goal,done = False,\
        num_pos=self.num_pos)
        
        return self.state.to_obs()

    def step(self, action):
        """
        The agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect. With probability 2/3, the actions
        cause the agent to move one cell in the corresponding direction, and with probability 1/3,
        the agent moves instead in one of the other three directions, each with 1/9 probability. In
        either case, if the movement would take the agent into a wall then the agent remains in the
        same cell.

        We consider a case in which rewards are zero on all state transitions.
        """
        if not self.open:
            raise Exception("Environment should be reseted")
        try:
            nextcell = tuple(self.currentcell + self.directions[action])
        except TypeError:
            nextcell = tuple(self.currentcell + self.directions[action[0]])

        if not self.occupancy[nextcell]:
            self.currentcell = nextcell
            if np.random.uniform() < 1/3:#impossible??
                empty_cells = self.empty_around(self.currentcell)
                self.currentcell = empty_cells[np.random.randint(len(empty_cells))]

        position_n = self.tostate[self.currentcell]

        self.state.current_steps += 1
        self.state.done = (position_n == self.state.goal_n) or (self.state.current_steps >= self.max_epilen)
        info = {}
        if self.state.done:
            # print(self.current_step, state == self.goal, self.max_epilen)
            info = {'episode': {'r': 10 - self.state.current_steps*0.1 if (position_n==self.state.goal_n)\
             else -self.state.current_steps*0.1,
                                'l': self.state.current_steps}}
            self.open=False
        self.state.position_n = position_n

        if position_n == self.state.goal_n:
            reward = 10
        else:
            reward = -0.1
        return self.state.to_obs(), reward, self.state.done, info

    def seed(self, seed):
        np.random.seed(seed)

    def close(self):
        self.open=False

    def render(self):
        raise NotImplementedError()
    
    def inner_state(self):
        return self.state

    def load(self,state):
        self.state=state
# class Fourrooms(FourroomsBase):
#     # metadata = {'render.modes':['human']}
#     # state :   number of state, counted from row and col
#     # cell : (i,j)
#     # observation : resultList[state]
#     # small : 104 large 461
#     def __init__(self, max_epilen=100, goal=None):
#         super().__init__(max_epilen,goal)
#         self.viewer = Viewer(self.block_size * len(self.occupancy), self.block_size * len(self.occupancy[0]))
#         self.blocks = self.make_blocks()

#     def make_blocks(self):
#         blocks = []
#         size = self.block_size
#         for i, row in enumerate(self.occupancy):
#             for j, o in enumerate(row):
#                 if o == 1:
#                     v = [[i * size, j * size], [i * size, (j + 1) * size], [(i + 1) * size, (j + 1) * size],
#                          [(i + 1) * size, (j) * size]]
#                     geom = make_polygon(v, filled=True)
#                     geom.set_color(0, 0, 0)
#                     blocks.append(geom)
#                     self.viewer.add_geom(geom)
#         return blocks

#     def check_state(self, states, info="None"):
#         assert all([int(ob)<self.num_pos for ob in states]), "what happened? " + info

#     def add_block(self, x, y, color):
#         size = self.block_size
#         v = [[x * size, y * size], [x * size, (y + 1) * size], [(x + 1) * size, (y + 1) * size],
#              [(x + 1) * size, y * size]]
#         geom = make_polygon(v, filled=True)
#         r, g, b = color
#         geom.set_color(r, g, b)
#         self.viewer.add_onetime(geom)

#     def render(self, mode=0):

#         if self.currentcell[0] > 0:
#             x, y = self.currentcell
#             # state = self.tostate[self.currentcell]
#             # self.add_block(x, y, tuple(self.rand_color[state]/255))
#             self.add_block(x, y, (0, 0, 1))

#         x, y = self.tocell[self.goal]
#         self.add_block(x, y, (1, 0, 0))
#         # self.viewer.
#         arr = self.viewer.render(return_rgb_array=True)

#         return arr
#     def all_states(self):
#         return list(range(self.observation_space.n))


class FourroomsNorender(FourroomsBase):
    """
    A rendered version.
    Image :(104,104,3)
    """
    def __init__(self, max_epilen=100, goal=77,seed=0):
        super().__init__(max_epilen,goal,seed)
        self.wall_blocks = self.make_wall_blocks()
        #render origin wall blocks to speed up rendering
        self.origin_background = self.render_with_blocks(
            255 * np.ones((self.obs_height, self.obs_width, 3), dtype=np.uint8),
            self.wall_blocks)
        self.agent_color = np.random.rand(100, 3)
        # print(self.background.shape)

    def render(self, mode=0):
        """
        render currentcell\walls\background,you can add blocks by parameter.
        Render mode is reserved for 
        """
        #render agent
        blocks=(self.make_basic_blocks())

        arr = self.render_with_blocks(self.origin_background, blocks)

        return arr
    def make_basic_blocks(self):
        blocks=[]
        if self.currentcell[0] > 0:
            x, y = self.currentcell
            #blocks.append(self.make_block(x, y, self.agent_color[np.random.randint(100)]))
            blocks.append(self.make_block(x, y, (0, 0, 1)))
        #render goal
        x, y = self.tocell[self.state.goal_n]
        blocks.append(self.make_block(x, y, (1, 0, 0)))
        return blocks

    def render_with_blocks(self, background, blocks)->np.array:
        background = np.copy(np.array(background))
        assert background.shape[-1] == len(background.shape) == 3, background.shape
        for block in blocks:
            v, color = block
            color = np.array(color).reshape(-1) * 255
            background[v[0][0]:v[2][0], v[0][1]:v[2][1], :] = color.astype(np.uint8)
        # assert background.shape[-1] == len(background.shape) == 3,background.shape
        return background

    def make_wall_blocks(self):

        blocks = []
        size = self.block_size
        for i, row in enumerate(self.occupancy):
            for j, o in enumerate(row):
                if o == 1:
                    v = [[i * size, j * size], [i * size, (j + 1) * size], [(i + 1) * size, (j + 1) * size],
                         [(i + 1) * size, (j) * size]]
                    color = (0, 0, 0)
                    geom = (v, color)
                    blocks.append(geom)
        return blocks

    def make_block(self, x, y, color):
        """
        color in [0,1]
        """
        size = self.block_size
        v = [[x * size, y * size], [x * size, (y + 1) * size], [(x + 1) * size, (y + 1) * size],
             [(x + 1) * size, y * size]]
        geom = (v, color)
        return geom
# register(
#     id='Fourrooms-v0',
#     entry_point='fourrooms:Fourrooms',
#     timestep_limit=20000,
#     reward_threshold=1,
# )

def check_render(env):
    env.reset()
    cv2.imwrite('test0.jpg',env.render())
    env.step(0)
    cv2.imwrite('test1.jpg',env.render())
    env.step(1)
    cv2.imwrite('test2.jpg',env.render())
    env.step(2)
    cv2.imwrite('test3.jpg',env.render())
    env.step(3)
    cv2.imwrite('test4.jpg',env.render())

def check_run(env):
    reward_list=[]
    for i in range(1000):
        obs,reward,done,_=env.step(env.action_space.sample())
        reward_list.append(reward)
        if done:
            env.reset()
            #print("i={},done\n".format(i))
            #print("reward is: "+str(np.sum(reward_list))+'\n')
            reward_list=[]

if __name__=='__main__':
    #basic test
    
    env_origin=ImageInputWarpper(FourroomsNorender())
    check_render(env_origin)
    check_env(env_origin,warn=True)
    check_run(env_origin)

    # stable-baseline test
    # NOTE:well-trained in 100k timesteps by ACKTR for block_size=3

    env = make_vec_env(lambda: env_origin, n_envs=1)
    model = ACKTR('CnnPolicy',env_origin, verbose=1)
    model.learn(total_timesteps=3000)
    print("Stable_baseline evaluation starts.....\n")
    #NOTE:evaluate_policy needs vec_env
    reward_mean,reward_std=evaluate_policy(model,env,n_eval_episodes=20,deterministic=False)

    print("mean reward:"+str(reward_mean)+'\n')
    print("reward std:"+str(reward_std)+'\n')
    print("custom evaluation begin\n")

    env=ImageInputWarpper(FourroomsNorender())
    obs = env.reset()
    reward_list_total=[]
    epilen_list=[]
    reward_list=[]
    last_end=0
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        reward_list.append(rewards)
        if dones:
            obs=env.reset()
            epilen_list.append(i-last_end)
            last_end=i
            reward_list_total.append(np.sum(reward_list))
            reward_list=[]
            if i>900:
                break
    print("mean reward:{}\n".format(np.mean(reward_list_total)))
    print("mean epilen:{}\n".format(np.mean(epilen_list)))