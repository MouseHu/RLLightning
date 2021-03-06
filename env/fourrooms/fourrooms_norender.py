import numpy as np
import gym
import time
from gym import error, spaces
from gym import core, spaces
from gym.envs.registration import register
import random
from env.rendering import *
from copy import copy


class Fourrooms(object):
    # metadata = {'render.modes':['human']}
    # state :   number of state, counted from row and col
    # cell : (i,j)
    # observation : resultList[state]
    # small : 104 large 461
    def __init__(self, max_epilen=100):
        self.init_basic(max_epilen)
        self.viewer = Viewer(self.block_size * len(self.occupancy), self.block_size * len(self.occupancy[0]))
        self.blocks = self.make_blocks()

    def init_basic(self, max_epilen):
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
        self.block_size = 8
        self.occupancy = np.array(
            [list(map(lambda c: 1 if c == '1' else 0, line)) for line in self.layout.splitlines()])
        self.num_pos = int(np.sum(self.occupancy == 0))
        self.obs_height = self.block_size * len(self.occupancy)
        self.obs_width = self.block_size * len(self.occupancy[0])
        # From any state the agent can perform one of four actions, up, down, left or right
        self.action_space = spaces.Discrete(4)
        # self.observation_space = spaces.Discrete(np.sum(self.occupancy == 0))
        self.observation_space = spaces.Discrete(self.num_pos)

        self.directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1)), np.array((0, 1))]
        # self.rng = np.random.RandomState(1234)

        self.rand_color = np.random.randint(0, 255, (200, 3))
        self.tostate = {}
        self.semantics = dict()
        statenum = 0
        # print("Here", len(self.occupancy), len(self.occupancy[0]))
        for i in range(len(self.occupancy)):
            for j in range(len(self.occupancy[0])):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i, j)] = statenum
                    statenum += 1
        self.tocell = {v: k for k, v in self.tostate.items()}

        self.goal = 62

        self.init_states = list(range(self.observation_space.n))
        self.init_states.remove(self.goal)
        # random encode
        self.mapping = np.arange(self.num_pos)
        self.dict = np.zeros((self.observation_space.n, 3))
        self.Row = np.shape(self.occupancy)[0]
        self.Col = np.shape(self.occupancy)[1]
        self.current_steps = 0
        self.max_epilen = max_epilen
        self.get_dict()
        self.currentcell = (-1, -1)
        self.reward_range = (0, 1)
        self.metadata = None
        self.done = False
        self.allow_early_resets = True
        self.unwrapped = self
        self.state = -1

    def make_blocks(self):
        blocks = []
        size = self.block_size
        for i, row in enumerate(self.occupancy):
            for j, o in enumerate(row):
                if o == 1:
                    v = [[i * size, j * size], [i * size, (j + 1) * size], [(i + 1) * size, (j + 1) * size],
                         [(i + 1) * size, (j) * size]]
                    geom = make_polygon(v, filled=True)
                    geom.set_color(0, 0, 0)
                    blocks.append(geom)
                    self.viewer.add_geom(geom)
        return blocks

    def check_obs(self, obs, info="None"):
        # print([ob for ob in obs if ob not in self.mapping])
        assert all([int(ob) in self.mapping for ob in obs]), "what happened? " + info

    def empty_around(self, cell):
        avail = []
        for action in range(self.action_space.n):
            nextcell = tuple(cell + self.directions[action])
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
        return avail

    def reset(self, state=-1):
        # state = self.rng.choice(self.init_states)
        # self.viewer.close()
        if state < 0:
            state = np.random.choice(self.init_states)
        self.currentcell = self.tocell[state]
        self.done = False
        self.current_steps = 0
        self.state = state
        return np.array(self.mapping[state])

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

        # print(self.currentcell, self.directions, action)
        try:
            nextcell = tuple(self.currentcell + self.directions[action])
        except TypeError:
            nextcell = tuple(self.currentcell + self.directions[action[0]])

        if not self.occupancy[nextcell]:
            self.currentcell = nextcell
            if np.random.uniform() < 0.:
                # if self.rng.uniform() < 1/3.:
                empty_cells = self.empty_around(self.currentcell)
                # self.currentcell = empty_cells[self.rng.randint(len(empty_cells))]
                self.currentcell = empty_cells[np.random.randint(len(empty_cells))]

        state = self.tostate[self.currentcell]

        self.current_steps += 1
        self.done = state == self.goal or self.current_steps >= self.max_epilen
        # if self.current_steps >= self.max_epilen:
        #     self.done = True
        info = {}
        if self.done:
            # print(self.current_step, state == self.goal, self.max_epilen)
            info = {'episode': {'r': 100 - self.current_steps if state == self.goal else -self.current_steps,
                                'l': self.current_steps}}
        # print(self.currentcell)
        self.state = state
        if state == self.goal:
            reward = 100
        else:
            reward = -1

        return np.array(self.mapping[state]), reward, self.done, info

    def get_dict(self):
        count = 0
        for i in range(self.Row):
            for j in range(self.Col):
                if self.occupancy[i, j] == 0:
                    # code
                    self.dict[count, 0] = self.mapping[count]
                    # i,j
                    self.dict[count, 1] = i
                    self.dict[count, 2] = j

                    self.semantics[self.mapping[count]] = str(i) + '_' + str(j)
                    count += 1

        # print(self.semantics)
        return self.semantics

    def add_block(self, x, y, color):
        size = self.block_size
        v = [[x * size, y * size], [x * size, (y + 1) * size], [(x + 1) * size, (y + 1) * size],
             [(x + 1) * size, y * size]]
        geom = make_polygon(v, filled=True)
        r, g, b = color
        geom.set_color(r, g, b)
        self.viewer.add_onetime(geom)

    def render(self, mode=0):

        if self.currentcell[0] > 0:
            x, y = self.currentcell
            # state = self.tostate[self.currentcell]
            # self.add_block(x, y, tuple(self.rand_color[state]/255))
            self.add_block(x, y, (0, 0, 1))

        x, y = self.tocell[self.goal]
        self.add_block(x, y, (1, 0, 0))
        # self.viewer.
        arr = self.viewer.render(return_rgb_array=True)

        return arr

    def seed(self, seed):
        pass

    def close(self):
        pass

    def all_states(self):
        return self.mapping


class FourroomsNorender(Fourrooms):
    def __init__(self, max_epilen=100):
        self.init_basic(max_epilen)
        self.blocks = self.make_blocks()
        self.background = self.render_with_blocks(np.zeros((self.obs_height, self.obs_width, 3), dtype=np.uint8),
                                                  self.blocks)

    def render(self, mode=0):
        blocks = []
        if self.currentcell[0] > 0:
            x, y = self.currentcell
            # state = self.tostate[self.currentcell]
            # self.add_block(x, y, tuple(self.rand_color[state]/255))
            blocks.append(self.make_block(x, y, (0, 0, 1)))

        x, y = self.tocell[self.goal]
        blocks.append(self.make_block(x, y, (1, 0, 0)))
        # self.add_block(x, y, (1, 0, 0))
        # self.viewer.
        arr = self.render_with_blocks(self.background, blocks)

        return arr

    def render_with_blocks(self, background, blocks):
        background = np.copy(np.array(background))
        assert background.shape[-1] == len(background.shape) == 3
        for block in blocks:
            v, color = block
            background[v[0, 0]:v[2, 0], v[0, 1]:v[2, 1], :] = np.array(color)
        return background

    def make_blocks(self):
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
