import argparse
from argparse import Namespace
from torch.optim.optimizer import Optimizer

import torch
from typing import Tuple,List
import torch.optim as optim

from algorithm.base_learner import BaseLearner


class DQNLearner(BaseLearner):
    def __init__(self, args: Namespace, components: Namespace) -> None:
        super().__init__(args, components)
        self.hparams = args
        print("Warming up ...")
        self.populate(self.args.warmup)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], nb_batch):
        # Calculates training loss
        loss, train_info = self.agent.compute_loss(batch)

        self.rollout(num_step=self.args.update_freq)

        if self.global_step % self.args.target_freq == 0:
            self.agent.update_target()

        if self.num_steps % self.args.eval_freq == 0:
            self.evaluate(num_episode=self.args.eval_episodes)

        if self.global_step % self.args.log_freq == 0:
            for info_name, info_value in train_info.items():
                self.log("losses/{}".format(info_name), info_value)

        return loss

    def explore_schedule(self, num_steps):
        return max(self.args.eps_end, self.args.eps_start - (num_steps + 0.0) / self.args.eps_last_frame)

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer"""
        optimizer = optim.Adam(self.agent.parameters(), lr=self.args.lr)
        return [optimizer]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--target_freq", type=int, default=250,
                            help="how many frames do we update the target network")
        parser.add_argument("--update_freq", type=int, default=4, help="how many frames do we train our network")
        parser.add_argument("--eps_last_frame", type=int, default=100000,
                            help="what frame should epsilon stop decaying")
        parser.add_argument("--eps_start", type=float, default=1.0, help="starting value of epsilon")
        parser.add_argument("--eps_end", type=float, default=0.01, help="final value of epsilon")
        parser.add_argument("--max_test_step", type=int, default=2000, help="max steps for testing")
        parser.add_argument("--eval_freq", type=int, default=2500, help="max steps for testing")
        parser.add_argument("--eval_episodes", type=int, default=5, help="max episodes for testing")

        return parser
