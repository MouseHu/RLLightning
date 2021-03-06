import argparse
from argparse import Namespace

import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from typing import Tuple,List

from algorithm.base_learner import BaseLearner


class ActorCriticLearner(BaseLearner):
    def __init__(self, args: Namespace, components: Namespace) -> None:
        super().__init__(args, components)
        self.hparams = args
        print("Warming up ...")
        self.populate(self.args.warmup)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], nb_batch: int, optimizer_idx: int):
        raise NotImplementedError

    def explore_schedule(self, num_steps):
        return 0

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer"""
        critic_optimizer = optim.Adam(self.agent.critic.parameters(), lr=self.args.critic_lr)
        actor_optimizer = optim.Adam(self.agent.actor.parameters(), lr=self.args.actor_lr)
        return [critic_optimizer,actor_optimizer]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--target_freq", type=int, default=100,
                            help="how many frames do we update the target network")
        parser.add_argument("--training_freq", type=int, default=100, help="how frequently do we train our network")
        parser.add_argument("--training_step", type=int, default=100,
                            help="how many step do we train our network")
        parser.add_argument("--max_test_step", type=int, default=5000, help="max steps for testing")
        parser.add_argument("--eval_freq", type=int, default=10000, help="evaluation freq")
        parser.add_argument("--tau", type=float, default=0.005, help="Polyak averaging coefficient")
        parser.add_argument("--explore_noise", type=float, default=0.1, help="Exloration std for gaussian noise")
        parser.add_argument("--noise_clip", type=float, default=0.2,
                            help="Action noise clip for target policy smoothing")
        return parser
