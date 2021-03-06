from argparse import Namespace

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from typing import List

from buffer.dataset import RLDataset


class AbstractLearner(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self._dataloader = None

    def rollout(self, num_step, train=True):
        raise NotImplementedError

    def populate(self, steps):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def configure_optimizers(self) -> List[Optimizer]:
        raise NotImplementedError

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

        self.state = None

        # basic logging
        self.num_steps = 0

    def get_state(self, train=True):
        state = self.agent.state if train else self.agent.eval_state
        if len(state.shape) > 1:  # image input
            state = state.astype(np.float) / 255.0
        state = torch.tensor([state], device=self.device)
        return state

    def explore_schedule(self, num_steps):
        raise NotImplementedError

    def rollout(self, num_step, train=True):
        # Rollout
        for i in range(num_step):
            self.num_steps += 1 if train else 0
            epsilon = self.explore_schedule(self.num_steps) if train else 0
            new_state, reward, done, info = self.agent.step(self.get_state(train), epsilon, train)
            if done:
                prefix = 'train/' if train else 'eval/'
                for k, v in info.items():
                    self.log(prefix + k, v, on_step=True, prog_bar='return' in k)
                self.log(prefix + 'steps', self.num_steps + (0 if train else i), prog_bar=True)

    def populate(self, steps: int = 1000) -> None:
        self.agent.reset(False)  # reset eval env
        self.agent.reset()
        for i in range(steps):
            self.num_steps += 1
            self.agent.step(self.get_state(), epsilon=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.agent.policy(x)
        return output

    def validation_step(self, *args, **kwargs):
        pass

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer"""
        optimizer = optim.Adam(self.agent.parameters(), lr=self.args.lr)
        return [optimizer]

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
