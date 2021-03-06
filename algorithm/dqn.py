import argparse
from argparse import Namespace

import torch
from typing import Tuple

from algorithm.base_learner import BaseLearner


class DQNLearner(BaseLearner):
    def __init__(self, args: Namespace, components: Namespace) -> None:
        super().__init__(args, components)
        self.hparams = args
        print("Warming up ...")
        self.populate(self.args.warmup)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], nb_batch):
        self.rollout(num_step=self.args.update_freq)
        # Calculates training loss
        loss, q_mean, q_target_mean = self.agent.compute_loss(batch)
        self.log("losses/loss", loss)
        self.log("losses/q_mean", q_mean)
        self.log("losses/q_target_mean", q_target_mean)

        # Soft update of target network
        if self.global_step % self.args.sync_rate == 0:
            self.agent.update_target()

        if self.num_steps % self.args.eval_freq == 0:
            self.rollout(num_step=self.args.max_test_step, train=False)
        return loss

    def explore_schedule(self, num_steps):
        return max(self.args.eps_end, self.args.eps_start - (num_steps + 0.0) / self.args.eps_last_frame)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--sync_rate", type=int, default=250,
                            help="how many frames do we update the target network")
        parser.add_argument("--update_freq", type=int, default=4, help="how many frames do we train our network")
        parser.add_argument("--eps_last_frame", type=int, default=100000,
                            help="what frame should epsilon stop decaying")
        parser.add_argument("--eps_start", type=float, default=1.0, help="starting value of epsilon")
        parser.add_argument("--eps_end", type=float, default=0.01, help="final value of epsilon")
        parser.add_argument("--max_test_step", type=int, default=2000, help="max steps for testing")
        parser.add_argument("--eval_freq", type=int, default=2500, help="max steps for testing")
        return parser
