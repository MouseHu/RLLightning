from torch.utils.data.dataset import IterableDataset, T_co
from buffer.replay_buffer import ReplayBuffer
from typing import Tuple


class RLDataset(IterableDataset):

    def __getitem__(self, index) -> T_co:
        pass

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 32) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Tuple:
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]


class RLDatasetMem(RLDataset):
    """
    RL Dataset used with episodic memory
    """

    def __iter__(self) -> Tuple:
        samples = self.buffer.sample(self.sample_size)
        states, actions, rewards, dones, new_states = \
            samples['obs0'], samples['actions'], samples['rewards'], samples['terminals1'], samples['obs1']
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]
