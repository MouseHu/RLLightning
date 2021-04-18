import numpy as np

from buffer.replay_buffer import ReplayBufferWrapper
from utils.segment_tree import SumSegmentTree, MinSegmentTree


class PrioritizedReplayBuffer(ReplayBufferWrapper):
    def __init__(self, base_buffer):
        """
        Create Prioritized Replay buffer.

        See Also ReplayBuffer.__init__

        :param size: (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories
            are dropped.
        :param alpha: (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        super().__init__(base_buffer)
        args = base_buffer.args
        assert args.alpha >= 0
        self._alpha = args.alpha

        it_capacity = 1
        while it_capacity < self.args.size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, obs_t, action, reward, obs_tp1, done):
        idxes = self.base_buffer.add(obs_t, action, reward, obs_tp1, done)
        if idxes is not None:
            self._it_sum[idxes] = self._max_priority ** self._alpha
            self._it_min[idxes] = self._max_priority ** self._alpha

    def extend(self, obs_t, action, reward, obs_tp1, done):
        idxes = self.base_buffer.extend(obs_t, action, reward, obs_tp1, done)
        for idx in idxes:
            self._it_sum[idx] = self._max_priority ** self._alpha
            self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        mass = []
        total = self._it_sum.sum(*self.sample_range)
        # TODO(szymon): should we ensure no repeats?
        mass = np.random.random(size=batch_size) * total
        idx = self._it_sum.find_prefixsum_idx(mass)
        return idx

    def sample(self, batch_size: int, beta: float = 1, env=None):
        assert beta > 0

        idxes = self._sample_proportional(batch_size)
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self.base_buffer)) ** (-beta)
        p_sample = self._it_sum[idxes] / self._it_sum.sum()
        weights = (p_sample * len(self.base_buffer)) ** (-beta) / max_weight
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(idxes) >= 0
        assert np.max(idxes) < len(self.base_buffer)
        self._it_sum[idxes] = priorities ** self._alpha
        self._it_min[idxes] = priorities ** self._alpha

        self._max_priority = max(self._max_priority, np.max(priorities))
