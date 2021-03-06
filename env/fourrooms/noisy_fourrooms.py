import numpy as np
import gym
from gym import spaces
# from attn_toy.env.fourrooms import Fourrooms as Fourrooms
from env.fourrooms import FourroomsNorender as Fourrooms


class ImageInputWarpper(gym.Wrapper):

    def __init__(self, env, max_steps=100):
        gym.Wrapper.__init__(self, env)
        screen_height = self.env.obs_height
        screen_width = self.env.obs_width
        self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        #print(self.observation_space)
        # self.num_steps = 0
        self.max_steps = max_steps
        # self.state_space_capacity = self.env.state_space_capacity
        self.mean_obs = None

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        # self.num_steps += 1
        if self.num_steps >= self.max_steps:
            done = True
        obs = self.env.render(state)
        #print(obs.shape)correct
        # print("step reporting",done)
        # if self.mean_obs is None:
        #     self.mean_obs = np.mean(obs)
        #     print("what is wrong?",self.mean_obs)
        # obs = obs - 0.5871700112336601
        # info['ori_obs'] = ori_obs
        info['s_tp1'] = state
        return obs, reward, done, info

    def reset(self, state=-1):
        if state < 0:
            state = np.random.randint(0, self.state_space_capacity)
        self.env.reset(state)
        # self.num_steps = self.env.num_steps
        obs = self.env.render(state)
        # print("reset reporting")
        # if self.mean_obs is None:
        #     self.mean_obs = np.mean(obs)
        # print("what is wrong? reset",self.mean_obs)
        # obs = obs - 0.5871700112336601
        # info['ori_obs'] = ori_obs
        return obs.astype(np.uint8)


class FourroomsDynamicNoise(Fourrooms):  # noise type = dynamic relevant
    def __init__(self, max_epilen=100, obs_size=128, seed=0, goal=77):
        np.random.seed(seed)
        super(FourroomsDynamicNoise, self).__init__(max_epilen, goal)
        self.obs_size = obs_size
        self.obs_height = obs_size
        self.obs_width = obs_size
        self.background = np.random.randint(0, 255, (10, 1, 1, 3))
        self.background[:, :, :, 2] = 0
        self.background = np.tile(self.background, (1, obs_size, obs_size, 1))
        self.seed = seed
        self.color = np.random.randint(100, 255, (200, 3))
        self.color[:, 2] = 100
        self.observation_space = spaces.Discrete(self.num_pos * 3)
        self.state_space_capacity = self.observation_space.n

    def render(self, state=-1):
        which_background = state // self.num_pos
        # obs = np.zeros((self.obs_size, self.obs_size, 3))
        # obs[:12, :12, :] = self.color[state + 1]

        # obs = np.random.randint(0, 255, (self.obs_size, self.obs_size, 3))
        obs = np.tile(self.color[which_background][np.newaxis, np.newaxis, :], (self.obs_size, self.obs_size, 1))
        # obs = (state+100) * np.ones((self.obs_size,self.obs_size))

        arr = super(FourroomsDynamicNoise, self).render(state)
        padding_height, padding_width = (obs.shape[0] - arr.shape[0]) // 2, (obs.shape[1] - arr.shape[1]) // 2
        obs[padding_height:padding_height + arr.shape[0], padding_width:padding_width + arr.shape[1], :] = arr
        return obs.astype(np.uint8)

    def step(self, action):
        state, reward, done, info = super(FourroomsDynamicNoise, self).step(action)
        state += self.num_pos * (self.num_steps % 3)
        return state, reward, done, info

    def reset(self, state=-1):
        obs = super(FourroomsDynamicNoise, self).reset(state % self.num_pos)
        self.num_steps = state % 3
        return state


class FourroomsDynamicNoise2(Fourrooms):  # noise type = state relevant
    def __init__(self, max_epilen=100, obs_size=128, seed=0, goal=77):
        np.random.seed(seed)
        super(FourroomsDynamicNoise2, self).__init__(max_epilen, goal)
        self.obs_size = obs_size
        self.obs_height = obs_size
        self.obs_width = obs_size
        self.background = np.random.randint(0, 255, (10, 1, 1, 3))
        self.background[:, :, :, 2] = 0
        self.background = np.tile(self.background, (1, obs_size, obs_size, 1))
        self.seed = seed
        self.color = np.random.randint(100, 255, (200, 3))
        self.color[:, 2] = 100
        self.observation_space = spaces.Discrete(self.num_pos * max_epilen)
        self.state_space_capacity = self.num_pos * max_epilen
        self.last_action = -1

    def step(self, action):
        state, reward, done, info = super(FourroomsDynamicNoise2, self).step(action)
        state += self.num_pos * self.num_steps
        return state, reward, done, info

    def reset(self, state=-1):
        self.state = state
        obs = super(FourroomsDynamicNoise2, self).reset(state % self.num_pos)
        self.num_steps = state // self.num_pos
        return state

    def render(self, state=-1):
        # which_background = self.num_steps % 3
        # obs = np.zeros((self.obs_size, self.obs_size, 3))
        # obs[:12, :12, :] = self.color[state + 1]
        obs = np.tile(self.color[self.num_steps + 1][np.newaxis, np.newaxis, :], (self.obs_size, self.obs_size, 1))
        # obs = np.random.randint(0, 255, (self.obs_size, self.obs_size, 3))
        # obs = np.tile(self.color[which_background][np.newaxis, np.newaxis, :], (self.obs_size, self.obs_size, 1))
        # obs = (state+100) * np.ones((self.obs_size,self.obs_size))

        arr = super(FourroomsDynamicNoise2, self).render(state % self.num_pos)
        padding_height, padding_width = (obs.shape[0] - arr.shape[0]) // 2, (obs.shape[1] - arr.shape[1]) // 2
        obs[padding_height:padding_height + arr.shape[0], padding_width:padding_width + arr.shape[1], :] = arr
        return obs.astype(np.uint8)


class FourroomsDynamicNoise3(Fourrooms):  # noise type = action relevant
    def __init__(self, max_epilen=100, obs_size=128, seed=0, goal=77):
        np.random.seed(seed)
        super(FourroomsDynamicNoise3, self).__init__(max_epilen, goal)
        self.agent_color = np.tile(np.array([[1, 0, 0]]), (100, 1))
        self.obs_size = obs_size
        self.obs_height = obs_size
        self.obs_width = obs_size
        self.background = np.random.randint(0, 255, (10, 1, 1, 3))
        self.background[:, :, :, 2] = 0
        self.background = np.tile(self.background, (1, obs_size, obs_size, 1))
        self.seed = seed
        self.color = np.random.randint(100, 255, (200, 3))
        self.color[:, 2] = 100
        self.observation_space = spaces.Discrete(self.num_pos * self.action_space.n)
        self.state_space_capacity = self.observation_space.n

    def render(self, state=-1):
        which_background = state // self.num_pos
        # obs = np.zeros((self.obs_size, self.obs_size, 3))
        # obs[:12, :12, :] = self.color[state + 1]
        # print(which_background, self.color[which_background])
        # obs = np.random.randint(0, 255, (self.obs_size, self.obs_size, 3))
        obs = np.tile(self.color[which_background][np.newaxis, np.newaxis, :], (self.obs_size, self.obs_size, 1))
        # obs = (state+100) * np.ones((self.obs_size,self.obs_size))

        arr = super(FourroomsDynamicNoise3, self).render(state)
        padding_height, padding_width = (obs.shape[0] - arr.shape[0]) // 2, (obs.shape[1] - arr.shape[1]) // 2
        obs[padding_height:padding_height + arr.shape[0], padding_width:padding_width + arr.shape[1], :] = arr
        return obs.astype(np.uint8)

    def step(self, action):
        state, reward, done, info = super(FourroomsDynamicNoise3, self).step(action)
        state += self.num_pos * action
        # print("state in step",state)
        return state, reward, done, info

    def reset(self, state=-1):
        obs = super(FourroomsDynamicNoise3, self).reset(state % self.num_pos)
        self.num_steps = state // self.num_pos
        return state


class FourroomsRandomNoise(Fourrooms):  # noise type = random
    def __init__(self, max_epilen=100, obs_size=128, seed=0, goal=77):
        np.random.seed(seed)
        super(FourroomsRandomNoise, self).__init__(max_epilen, goal)
        self.obs_size = obs_size
        self.obs_height = obs_size
        self.obs_width = obs_size
        # self.background = np.random.randint(0, 255, (10, 1, 1, 3))
        # self.background[:, :, :, 2] = 0
        # self.background = np.tile(self.background, (1, obs_size, obs_size, 1))
        self.random_background = np.random.randint(0, 255, (100, obs_size, obs_size, 3))
        # self.random_background[..., 2] = 100
        self.seed = seed
        self.color = np.random.randint(100, 255, (200, 3))
        # self.color[:, 2] = 100
        self.rand_range = 100
        self.observation_space = spaces.Discrete(self.num_pos * self.rand_range)
        self.state_space_capacity = self.observation_space.n
        self.which_background = -1

    def render(self, state=-1):
        which_background = state // self.num_pos
        # obs = np.zeros((self.obs_size, self.obs_size, 3))
        # obs[:12, :12, :] = self.color[state + 1]

        # obs = np.random.randint(0, 255, (self.obs_size, self.obs_size, 3))
        # obs = np.tile(self.color[which_background][np.newaxis, np.newaxis, :], (self.obs_size, self.obs_size, 1))
        obs = self.random_background[which_background]
        # obs = (state+100) * np.ones((self.obs_size,self.obs_size))

        arr = super(FourroomsRandomNoise, self).render(state % self.num_pos)
        padding_height, padding_width = (obs.shape[0] - arr.shape[0]) // 2, (obs.shape[1] - arr.shape[1]) // 2
        obs[padding_height:padding_height + arr.shape[0], padding_width:padding_width + arr.shape[1], :] = arr
        return obs.astype(np.uint8)

    def step(self, action):
        state, reward, done, info = super(FourroomsRandomNoise, self).step(action)
        self.which_background = np.random.randint(0, self.rand_range)
        state += self.num_pos * self.which_background
        return state, reward, done, info

    def reset(self, state=-1):
        self.which_background = np.random.randint(0, self.rand_range)
        super(FourroomsRandomNoise, self).reset(state % self.num_pos)
        return state


class FourroomsRandomNoisePos(FourroomsRandomNoise):  # noise type = random
    def __init__(self, max_epilen=100, obs_size=128, seed=0, goal=77):
        super(FourroomsRandomNoisePos, self).__init__(max_epilen, obs_size, seed, goal)

    def render(self, state=-1):
        obs = np.zeros((self.obs_height, self.obs_width, 3))
        pos = state // self.num_pos
        obs[pos * 12:pos * 12 + 12, :12] = self.color[pos]
        arr = super(FourroomsRandomNoise, self).render(state % self.num_pos)
        padding_height, padding_width = (obs.shape[0] - arr.shape[0]) // 2, (obs.shape[1] - arr.shape[1]) // 2
        obs[padding_height:padding_height + arr.shape[0], padding_width:padding_width + arr.shape[1], :] = arr
        return obs.astype(np.uint8)


class FourroomsOptimalNoise(Fourrooms):  # noise type = optimal action
    def __init__(self, max_epilen=100, obs_size=128, seed=0, goal=77, optimal_action=None):
        np.random.seed(seed)
        super(FourroomsOptimalNoise, self).__init__(max_epilen, goal)
        self.obs_size = obs_size
        self.obs_height = obs_size
        self.obs_width = obs_size
        # self.background = np.random.randint(0, 255, (10, 1, 1, 3))
        # self.background[:, :, :, 2] = 0
        # self.background = np.tile(self.background, (1, obs_size, obs_size, 1))
        self.seed = seed
        self.color = np.random.randint(0, 255, (200, 3))
        # self.color[:, 2] = 100
        self.observation_space = spaces.Discrete(self.num_pos)
        self.state_space_capacity = self.num_pos
        self.last_action = -1
        self.optimal_action = optimal_action

    def step(self, action):
        state, reward, done, info = super(FourroomsOptimalNoise, self).step(action)
        # state += self.num_pos * self.num_steps
        return state, reward, done, info

    def reset(self, state=-1):
        # self.num_steps = state // self.num_pos
        self.state = state
        obs = super(FourroomsOptimalNoise, self).reset(state % self.num_pos)
        return state

    def render(self, state=-1):
        # which_background = self.num_steps % 3
        # obs = np.zeros((self.obs_size, self.obs_size, 3))
        # obs[:12, :12, :] = self.color[state + 1]
        obs = np.tile(self.color[self.optimal_action[state]][np.newaxis, np.newaxis, :],
                      (self.obs_size, self.obs_size, 1))
        # obs = np.random.randint(0, 255, (self.obs_size, self.obs_size, 3))
        # obs = np.tile(self.color[which_background][np.newaxis, np.newaxis, :], (self.obs_size, self.obs_size, 1))
        # obs = (state+100) * np.ones((self.obs_size,self.obs_size))

        arr = super(FourroomsOptimalNoise, self).render(state % self.num_pos)
        padding_height, padding_width = (obs.shape[0] - arr.shape[0]) // 2, (obs.shape[1] - arr.shape[1]) // 2
        obs[padding_height:padding_height + arr.shape[0], padding_width:padding_width + arr.shape[1], :] = arr
        return obs.astype(np.uint8)


class FourroomsOptimalNoisePos(FourroomsOptimalNoise):  # noise type = optimal action
    def __init__(self, max_epilen=100, obs_size=128, seed=0, goal=77, optimal_action=None):
        super(FourroomsOptimalNoisePos, self).__init__(max_epilen, obs_size, seed, goal, optimal_action)

    def render(self, state=-1):
        obs = np.zeros((self.obs_height, self.obs_width, 3))
        pos = self.optimal_action[state]
        obs[pos * 12:pos * 12 + 12, :12] = self.color[pos]

        arr = Fourrooms.render(self,state % self.num_pos)
        padding_height, padding_width = (obs.shape[0] - arr.shape[0]) // 2, (obs.shape[1] - arr.shape[1]) // 2
        obs[padding_height:padding_height + arr.shape[0], padding_width:padding_width + arr.shape[1], :] = arr
        return obs.astype(np.uint8)


class FourroomsMyNoise(FourroomsOptimalNoise):  # noise type = optimal action
    def __init__(self, max_epilen=100, obs_size=128, seed=0, goal=77, optimal_action=None):
        super(FourroomsMyNoise, self).__init__(max_epilen, obs_size, seed, goal, optimal_action)

    # def render(self, state=-1):
    #     rnd = np.random.randint(0, self.noise_size)
    #     self.origin_background = self.render_with_blocks(self.noisy_background[rnd], self.blocks)
    #     return super(FourroomsMyNoise,self).render(state)

    def render(self, state=-1):
        obs = super(FourroomsMyNoise, self).render(state)
        # print(obs.shape)
        obs = obs[..., [2, 0, 1]]  # rgb -> brg
        return obs.astype(np.uint8)
