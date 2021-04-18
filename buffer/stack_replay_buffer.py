import os
import pickle as pkl

import numpy as np

from buffer.replay_buffer import ReplayBuffer


class StackReplayBuffer(ReplayBuffer):
    """
    Implementation of stacked buffer. Stack frames to reduce memory usage.
    """

    def __init__(self, args, component):
        super().__init__(args, component)
        buffer_size = int(args.buffer_size)

        self.curr_capacity = 0

        self.obs_space = component.env.observation_space
        self.action_shape = component.env.action_space.n
        self.frames = args.frames

        self.begin_obs = dict()
        self.frame_buffer = np.empty((buffer_size,) + self.obs_space.shape[1:], np.uint8)
        self.action_buffer = np.empty([buffer_size, 1], np.int)
        self.episode_steps = np.zeros((buffer_size,), np.int)
        self.reward_buffer = np.empty((buffer_size,), np.float32)
        self.done_buffer = np.empty((buffer_size,), np.bool)

        self.end_points = []
        self.next_id = -1 * np.ones(buffer_size)
        self.prev_id = [[] for _ in range(buffer_size)]

        self.sequence = []

    def __len__(self) -> int:
        return self.curr_capacity

    def _get_ring_obs(self, idx, num_frame):
        begin_idx = (idx + num_frame) % self._maxsize
        if begin_idx < idx:
            obs = np.concatenate([self.frame_buffer[idx:], self.frame_buffer[:begin_idx]], axis=0)
        else:
            obs = self.frame_buffer[idx:begin_idx, :, :]
        obs = obs[::-1, :, :]
        return obs

    def get_obs(self, idx):
        if self.episode_steps[idx] + 1 < self.frames:
            # need to append first
            first_idx = (idx + self.episode_steps[idx]) % self._maxsize
            try:
                begin_obs = self.begin_obs[first_idx][self.episode_steps[idx]:-1, :, :]
            except KeyError:
                print(idx, self.episode_steps[idx], first_idx)
                print(list(self.begin_obs.keys()))
                raise KeyError
            end_obs = self._get_ring_obs(idx, self.episode_steps[idx] + 1)
            obs = np.concatenate([begin_obs, end_obs], axis=0)
        else:
            obs = self._get_ring_obs(idx, self.frames)
        return obs

    def insert(self, obs, action, next_id=-1):
        index = self._next_idx
        self._next_idx = (self._next_idx + 1) % self._maxsize

        if self.curr_capacity >= self._maxsize:
            # Clean up old entry
            if index in self.end_points:
                self.end_points.remove(index)
                for prev_id in self.prev_id[index]:
                    if prev_id not in self.end_points:
                        self.end_points.append(prev_id)
            self.prev_id[index] = []
            self.next_id[index] = -1
            if self.episode_steps[index] == 0:
                del self.begin_obs[index]
            self.episode_steps[index] = 0
        else:
            self.curr_capacity = min(self._maxsize, self.curr_capacity + 1)
        # Store new entry
        self.frame_buffer[index] = obs[-1, :, :]
        self.action_buffer[index] = action

        if next_id >= 0:
            self.next_id[index] = next_id
            if index not in self.prev_id[next_id]:
                self.prev_id[next_id].append(index)

        return index

    def add_sequence(self, sequence):
        # print(sequence)
        next_id = -1
        episode_steps = len(sequence)
        ids = []
        for i, transition in enumerate(reversed(sequence)):
            obs, a, r, done = transition
            # print(np.mean(z))
            episode_steps = episode_steps - 1
            r = np.clip(r, -self._rews_scale, self._rews_scale)
            current_id = self.insert(obs, a, next_id)
            ids.append(current_id)
            if done:
                self.end_points.append(current_id)
            if episode_steps == 0:
                self.begin_obs[current_id] = obs
            self.frame_buffer[current_id] = obs[-1, :, :]
            self.reward_buffer[current_id] = r
            self.done_buffer[current_id] = done
            self.episode_steps[current_id] = episode_steps
            next_id = int(current_id)

        # for i, transition in enumerate(reversed(sequence)):
        #     obs, a, r, done = transition
        #     diff = obs - self.get_obs([ids[i]])[0]
        #     assert np.sum(abs(diff)) == 0, np.sum(abs(diff))

        return list(reversed(ids))

    def add(self, obs_t, action, reward, done, obs_tp1, end=None):
        self.sequence.append((obs_t, action, reward, done))
        end = done if end is None else end
        if end:
            self.sequence.append((obs_tp1, action, reward, done))
            idxes = self.add_sequence(self.sequence)
            self.sequence = []
            return idxes
        return

    def extend(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new batch of transitions to the buffer
        note that the sequence is truncated
        """
        self.sequence = []
        idxes = []
        for ind, data in enumerate(zip(obs_t, action, reward, obs_tp1, done)):
            o_t, a, r, o_tp1, d = data
            end = True if ind == len(obs_t) - 1 else None
            onetime_idxes = self.add(o_t, a, r, o_tp1, d, end)
            if onetime_idxes is not None:
                idxes += onetime_idxes
        return idxes

    def get_data(self, idx):
        idx_next = self.next_id[idx]
        return self.get_obs(idx), self.action_buffer[idx], self.reward_buffer[idx], self.get_obs(idx_next), \
               self.done_buffer[idx]

    def get_next_id(self, idx):
        assert 0 <= idx < len(self)
        return self.next_id[idx]

    def clean(self):
        buffer_size, obs_space, action_shape = self._maxsize, self.obs_space, self.action_shape
        self.curr_capacity = 0
        self._next_idx = 0

        self.frame_buffer = np.empty((buffer_size,) + obs_space.shape[:-1], np.float32)
        self.begin_obs = dict()
        self.action_buffer = np.empty((buffer_size,) + action_shape, np.uint8)
        self.reward_buffer = np.empty((buffer_size,), np.float32)
        self.done_buffer = np.empty((buffer_size,), np.bool)

    def save(self, file_dir):
        save_dict = {"frame_buffer": self.frame_buffer, "begin_obs": self.begin_obs,
                     "reward_buffer": self.reward_buffer, "done_buffer": self.done_buffer,
                     "curr_capacity": self.curr_capacity, "max_size": self._maxsize}
        with open(os.path.join(file_dir, "replay_buffer.pkl"), "wb") as memory_file:
            pkl.dump(save_dict, memory_file)
