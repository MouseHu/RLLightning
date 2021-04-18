import os
import pickle as pkl
import random

import numpy as np
from typing import List, Union


class ReplayBuffer(object):
    def __init__(self, args, component):
        self.args = args
        self.component = component

        self._storage = []
        self._maxsize = args.buffer_size
        self._rews_scale = args.rews_scale
        self._next_idx = 0

    def __len__(self) -> int:
        return len(self._storage)

    @property
    def storage(self):
        return self._storage

    @property
    def buffer_size(self) -> int:
        """float: Max capacity of the buffer"""
        return self._maxsize

    def can_sample(self, n_samples: int) -> bool:
        return len(self) >= n_samples

    def is_full(self) -> int:
        return len(self) == self.buffer_size

    def add(self, obs_t, action, reward, done, obs_tp1):
        data = (obs_t, action, reward, done, obs_tp1)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        cur_place = self._next_idx
        self._next_idx = (self._next_idx + 1) % self._maxsize
        return cur_place

    def extend(self, obs_t, action, reward, obs_tp1, done):
        idxes = []
        for data in zip(obs_t, action, reward, obs_tp1, done):
            if self._next_idx >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[self._next_idx] = data
            idxes.append(self._next_idx)
            self._next_idx = (self._next_idx + 1) % self._maxsize
        return idxes

    def get_next_id(self, idx):
        return (idx + 1) % len(self)

    def get_data(self, idx):
        return self._storage[idx]

    def _encode_sample(self, idxes: Union[List[int], np.ndarray]):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for idx in idxes:
            data = self.get_data(idx)
            obs_t, action, reward, done, obs_tp1 = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)

        if len(np.array(obses_t).shape) > 2:
            # image input
            obses_t = np.array(obses_t).astype(np.float32) / 255.0
            obses_tp1 = np.array(obses_tp1).astype(np.float32) / 255.0

        return (np.array(obses_t),
                np.array(actions),
                np.array(rewards),
                np.array(dones),
                np.array(obses_tp1),
                )

    @property
    def sample_range(self):
        # return the range of sample. It is useful when n-step is used
        return 0, len(self) - 1

    def sample(self, batch_size: int, **_kwargs):
        idxes = [random.randint(*self.sample_range) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def clean(self):
        del self._storage
        self._storage = []
        self._next_idx = 0

    def save(self, file_dir):
        save_dict = {"storage": self._storage, "max_size": self._maxsize}
        with open(os.path.join(file_dir, "replay_buffer.pkl"), "wb") as memory_file:
            pkl.dump(save_dict, memory_file)


class ReplayBufferWrapper(object):
    def __init__(self, base_buffer: ReplayBuffer):
        self.base_buffer = base_buffer

    def __getattr__(self, name):
        try:
            return getattr(self.base_buffer, name)
        except AttributeError:
            return getattr(self, name)
