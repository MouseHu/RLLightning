import cv2
import gym
import numpy as np

from utils.func_utils import remove_color


class VanillaEnv():
    def __init__(self, args):
        self.args = args
        if args.sticky:
            # frameskip is deterministic
            self.env = gym.make(args.env_name + 'Deterministic-v0').env
        else:
            self.env = gym.make(args.env_name + 'Deterministic-v4').env

        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(args.frames, 84, 84), dtype=np.float32)
        assert type(self.action_space) is gym.spaces.discrete.Discrete
        self.acts_dims = [self.action_space.n]
        self.obs_dims = list(self.observation_space.shape)

        self.render = self.env.render

        self.reset()
        self.num_episodes = 0
        self.total_steps = 0
        self.env_info = {
            'epi_length': self.process_info_steps,  # episode steps
            'epi_returns': self.process_info_rewards,  # episode cumulative rewards
            'num_episodes': self.process_info_num_episodes,
            'env_steps': self.process_info_total_steps,
        }

    def get_new_frame(self):
        # standard wrapper for atari
        frame = self.env._get_obs().astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        self.last_frame = frame.copy()
        return frame

    def get_obs(self):
        return self.last_obs.copy()

    def get_frame(self):
        return self.last_frame.copy()

    def process_info_steps(self, obs, reward, done, info):
        self.steps += 1
        return self.steps

    def process_info_rewards(self, obs, reward, done, info):
        self.rewards += reward
        return self.rewards

    def process_info_total_steps(self, obs, reward, done, info):
        self.total_steps += 1
        return self.total_steps

    def process_info_num_episodes(self, obs, reward, done, info):
        if done:
            self.num_episodes += 1
        return self.num_episodes

    def process_info(self, obs, reward, done, info):
        return {
            remove_color(key): value_func(obs, reward, done, info)
            for key, value_func in self.env_info.items()
        }

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info = self.process_info(obs, reward, done, info)
        self.frames_stack = self.frames_stack[1:] + [self.get_new_frame()]
        self.last_obs = np.stack(self.frames_stack, axis=0)
        # if self.steps == self.args.test_timesteps: done = True
        if not done:
            info = {}
        return self.last_obs.copy(), reward, done, info

    def reset_ep(self):
        self.steps = 0
        self.rewards = 0.0

    def reset(self):
        self.reset_ep()
        while True:
            flag = True
            self.env.reset()
            for _ in range(max(self.args.noop - self.args.frames, 0)):
                _, _, done, _ = self.env.step(0)
                if done:
                    flag = False
                    break
            if flag: break

        self.frames_stack = []
        for _ in range(self.args.frames):
            self.env.step(0)
            self.frames_stack.append(self.get_new_frame())

        self.last_obs = np.stack(self.frames_stack, axis=0)
        return self.last_obs.copy()
