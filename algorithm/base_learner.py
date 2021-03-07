from argparse import Namespace

import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from typing import List

from buffer.dataset import RLDataset
from utils.os_utils import merge_dicts


class AbstractLearner(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self._dataloader = None

    def rollout(self, num_step):
        raise NotImplementedError

    def evaluate(self, num_episodes):
        raise NotImplementedError

    def populate(self, steps):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def configure_optimizers(self) -> List[Optimizer]:
        raise NotImplementedError

    def validation_step(self, *args, **kwargs):
        pass

    @property
    def dataloader(self):
        if self._dataloader is None:
            self._dataloader = self.get_dataloader()
        return self._dataloader

    def get_dataloader(self) -> DataLoader:
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self.dataloader

    def val_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self.dataloader


class BaseLearner(AbstractLearner):
    def __init__(self, args: Namespace, components: Namespace) -> None:
        super().__init__()

        self.env = components.env
        self.agent = components.agent
        self.buffer = components.buffer
        self.args = args
        self.components = components
        self.state = None

        # basic logging
        self.num_steps = 0

    def get_state(self, train=True):
        state = self.agent.state if train else self.agent.eval_state
        if len(state.shape) > 1:  # image input
            state = state.astype(np.float32) / 255.0
        state = torch.tensor([state], device=self.device, dtype=torch.float32)
        return state

    def explore_schedule(self, num_steps):
        raise NotImplementedError

    def rollout(self, num_step):
        # Rollout
        for i in range(num_step):
            self.num_steps += 1
            epsilon = self.explore_schedule(self.num_steps)
            new_state, reward, done, info = self.agent.step(self.get_state(), epsilon, train=True)
            if done and self.num_steps % self.args.log_freq == 0:
                prefix = 'train/'
                for k, v in info.items():
                    if not isinstance(v, dict):  # dict value is temporally removed, it can be added in the future
                        self.log(prefix + k, v, on_step=True, prog_bar='epi_returns' in k)
                self.log(prefix + 'steps', self.num_steps, prog_bar=True)

    def evaluate(self, num_episode):
        infos = []
        episodes = 0
        while episodes < num_episode:
            new_state, reward, done, info = self.agent.step(self.get_state(train=False), 0, train=False)
            if done:
                episodes += 1
                infos.append(info)
        prefix = 'eval/'
        merged_info = merge_dicts(infos)
        for k, v in merged_info.items():
            self.log(prefix + k, v, on_step=True, prog_bar='epi_returns' in k)
        self.log(prefix + 'steps', self.num_steps, prog_bar=True)

    def populate(self, steps: int = 1000) -> None:
        self.agent.reset(False)  # reset eval env
        self.agent.reset()
        for i in range(steps):
            self.num_steps += 1
            self.agent.step(self.get_state(), epsilon=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.agent.policy(x)
        return output

    def configure_optimizers(self) -> List[Optimizer]:
        raise NotImplementedError

    def get_dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = RLDataset(self.buffer, self.args.batch_size)
        dataloader = DataLoader(
            # num_workers=8,
            dataset=dataset,
            batch_size=self.args.batch_size,
            sampler=None,
        )
        return dataloader
