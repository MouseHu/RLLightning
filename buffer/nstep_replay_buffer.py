from buffer.replay_buffer import *
import numpy as np


class NStepReplayBuffer(ReplayBuffer):
    def __init__(self, args, component):
        super(NStepReplayBuffer, self).__init__(args, component)
        self.num_step = args.num_step
        self.gamma = args.gamma

    def _encode_sample(self, idxes: Union[List[int], np.ndarray]):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            reward = 0
            for step in range(self.num_step):
                data = self._storage[(i + step) % self.buffer_size]
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

    def sample(self, batch_size: int, **_kwargs):
        """
        Sample a batch of experiences.

        :param batch_size: (int) How many transitions to sample.
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
        """

        idxes = [random.randint(0, len(self._storage) - self.num_step) for _ in range(batch_size)]
        return self._encode_sample(idxes)