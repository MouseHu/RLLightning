import gym
import numpy as np
from gym import spaces
from gym.core import Wrapper
from gym.spaces import Box


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class ClipActionsWrapper(gym.Wrapper):
    def step(self, action):
        import numpy as np
        action = np.nan_to_num(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ImageInputWarpper(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        screen_height = self.env.obs_height
        screen_width = self.env.obs_width
        self.observation_space = spaces.Box(low=0, high=255, shape= \
            (screen_height, screen_width, 3), dtype=np.uint8)
        # print(self.observation_space)
        self.mean_obs = None

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        obs = self.env.render()
        # print(obs.shape)correct
        # print("step reporting",done)
        # if self.mean_obs is None:
        #     self.mean_obs = np.mean(obs)
        #     print("what is wrong?",self.mean_obs)
        # obs = obs - 0.5871700112336601
        # info['ori_obs'] = ori_obs
        info['s_tp1'] = state
        return obs.astype(np.uint8), reward, done, info

    def reset(self):
        self.env.reset()
        obs = self.env.render()
        # print("reset reporting")
        # if self.mean_obs is None:
        #     self.mean_obs = np.mean(obs)
        # print("what is wrong? reset",self.mean_obs)
        # obs = obs - 0.5871700112336601
        # info['ori_obs'] = ori_obs
        return obs.astype(np.uint8)


class NHWCWrapper(Wrapper):
    def __init__(self, env):
        super(NHWCWrapper, self).__init__(env)

        obs_space = env.observation_space
        assert isinstance(obs_space, Box)
        low, high, shape = obs_space.low, obs_space.high, obs_space.shape
        # print("www",low,high,shape)
        new_shape = shape[1:] + (shape[0],)
        low = low.transpose((1, 2, 0))
        high = high.transpose((1, 2, 0))
        self.observation_space = Box(low, high, shape=new_shape)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        new_obs = obs.transpose((1, 2, 0))
        return new_obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        new_obs = obs.transpose((1, 2, 0))
        return new_obs


class TimestepWrapper(Wrapper):
    def __init__(self, env, scale=0.01):
        super(TimestepWrapper, self).__init__(env)
        self.scale = scale
        # low = np.append(self.env.observation_space.low, np.array([-np.inf]))
        # high = np.append(self.env.observation_space.high, np.array([np.inf]))
        # self.observation_space = gym.spaces.Box(low, high)
        self.time_step = 0
        # self.max_step = env.unwrapped.spec.max_episode_steps
        # print("max_step: ", self.max_step)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.time_step += 1
        # obs = np.append(obs, np.array(self.time_step * self.scale))
        # if done and self.time_step < self.max_step:
        #     truly_done = True
        # else:
        #     truly_done = False
        # info['truly_done'] = truly_done
        return obs, reward, done, info

    def reset(self):
        self.time_step = 0
        obs = self.env.reset()
        # obs = np.append(obs, np.array(self.time_step * self.scale))
        return obs


class EpisodicRewardWrapper(Wrapper):
    def __init__(self, env):
        super(EpisodicRewardWrapper, self).__init__(env)
        self.cum_reward = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.cum_reward += reward
        reward = self.cum_reward if done else 0
        return obs, reward, done, info

    def reset(self):
        self.cum_reward = 0
        return self.env.reset()


class MonitorWrapper(Wrapper):
    def __init__(self, env, gamma=0.99):
        super(MonitorWrapper, self).__init__(env)
        self.gamma = gamma
        self.episode_reward = 0
        self.discount_episode_reward = 0
        self.episode_length = 0
        self.num_episodes = 0
        self.max_step = env.unwrapped.spec.max_episode_steps
        print("max_step: ", self.max_step)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.episode_reward += reward
        self.discount_episode_reward += reward * (self.gamma ** self.episode_length)
        self.episode_length += 1
        if done:
            self.num_episodes += 1
            info['epi_length'] = self.episode_length
            info['epi_returns'] = self.episode_reward
            info['epi_discount_returns'] = self.discount_episode_reward
            info['num_episodes'] = self.num_episodes
        if done and self.episode_length < self.max_step:
            truly_done = True
        else:
            truly_done = False
        info['truly_done'] = truly_done
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.episode_length = self.episode_reward = self.discount_episode_reward = 0
        return self.env.reset()


class DelayedRewardWrapper(Wrapper):
    def __init__(self, env, delay=10):
        super(DelayedRewardWrapper, self).__init__(env)
        self.cum_reward = 0
        self.delay = delay
        self.time_step = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.cum_reward += reward
        self.time_step += 1
        if done or self.delay == 0 or (self.time_step % self.delay == 0):
            reward = self.cum_reward
            self.cum_reward = 0
        else:
            reward = 0

        return obs, reward, done, info

    def reset(self):
        self.cum_reward = 0
        self.time_step = 0
        return self.env.reset()

class ScaledWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env.env)# here assume env is RamEnv
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0