from torch.utils.data.dataset import IterableDataset, T_co
from typing import Iterable
from typing import Tuple

from buffer.replay_buffer import ReplayBuffer
from utils.func_utils import array_min2d


class RLDataset(IterableDataset):

    def __getitem__(self, index) -> T_co:
        pass

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 32) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Tuple:
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        states = array_min2d(states)
        actions = array_min2d(actions)
        rewards = array_min2d(rewards)
        dones = array_min2d(dones)
        new_states = array_min2d(new_states)
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


class ExperienceSourceDataset(IterableDataset):
    """
    Implementation from PyTorch Lightning Bolts:
    https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/datamodules/experience_source.py
    Basic experience source dataset. Takes a generate_batch function that returns an iterator.
    The logic for the experience source and how the batch is generated is defined the Lightning model itself
    """

    def __getitem__(self, index) -> T_co:
        pass

    def __init__(self, runner):
        self.runner = runner

    def __iter__(self) -> Iterable:
        iterator = self.runner.generate_samples()
        return iterator
