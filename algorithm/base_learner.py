from argparse import Namespace

import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from typing import List
from numbers import Number
from buffer.dataset import RLDataset
from utils.func_utils import merge_dicts


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
    def __init__(self, args: Namespace, component: Namespace) -> None:
        super().__init__()

        self.env = component.env
        self.agent = component.agent
        self.buffer = component.buffer
        self.args = args
        self.component = component
        self.state = None

        # basic logging
        self.num_steps = 0



    def explore_schedule(self, num_steps):
        raise NotImplementedError

    def rollout(self, num_step):
        # Fill buffer
        # Rollout
        for i in range(num_step):
            self.num_steps += 1
            epsilon = self.explore_schedule(self.num_steps)
            new_state, reward, done, info = self.agent.step(epsilon, train=True)
            if done or self.num_steps % self.args.log_freq == 0:
                prefix = 'train/'
                for k, v in info.items():
                    if 'truncated' not in k and isinstance(v, Number):
                        # dict value is temporally removed, it can be added in the future
                        self.log(prefix + k, v, prog_bar='epi_returns' in k)
                self.log(prefix + 'steps', self.num_steps,)

    def evaluate(self, num_episode):
        infos = []
        episodes = 0
        #print(num_episode)
        i=0
        while episodes < num_episode:
            i+=1
            new_state, reward, done, info, *_ = self.agent.step( 0, train=False)
            #print("stepping {}".format(i))
            if done:
                #print("episode done")
                episodes += 1
                infos.append(info)
        
        prefix = 'eval/'
        merged_info = merge_dicts(infos)
        for k, v in merged_info.items():
            if 'truncated' not in k:
                self.log(prefix + k, v, prog_bar='epi_returns' in k)
        self.log(prefix + 'steps', self.num_steps)

    def populate(self, steps: int = 1000) -> None:
        self.agent.reset(train=False)  # reset eval env
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
