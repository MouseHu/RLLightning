from buffer.replay_buffer import *


class NStepReplayBuffer(ReplayBufferWrapper):
    def __init__(self, base_buffer):
        super(NStepReplayBuffer, self).__init__(base_buffer)
        args = base_buffer.args
        self.num_step = args.n_step
        self.gamma = args.gamma

    def _encode_sample(self, idxes: Union[List[int], np.ndarray]):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            reward = 0
            idx = i
            for step in range(self.num_step):
                data = self.get_data[idx]
                idx = self.get_next_id(idx)
                obs_t, action, r, obs_tp1, done = data
                reward += (self.gamma ** step) * r
                if step == 0:
                    obses_t.append(np.array(obs_t, copy=False))
                    actions.append(np.array(action, copy=False))
                if step == self.num_step - 1 or done:
                    rewards.append(reward)
                    obses_tp1.append(np.array(obs_tp1, copy=False))
                    dones.append(done)
                    break
        return (np.array(obses_t),
                np.array(actions),
                np.array(rewards),
                np.array(obses_tp1),
                np.array(dones))

    def can_sample(self, n_samples: int) -> bool:
        return len(self.base_buffer) >= n_samples + self.num_step

    @property
    def sample_range(self):
        return 0, len(self.base_buffer) - self.num_step
