import argparse
from argparse import Namespace

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from typing import List

from algorithm.base_learner import BaseLearner
from buffer.dataset import ExperienceSourceDataset
from utils.func_utils import merge_dicts


class PPOLearner(BaseLearner):
    """
    PyTorch Lightning implementation of PPO.
    """

    def __init__(self, args: Namespace, component: Namespace) -> None:
        super().__init__(args, component)
        self.hparams = args
        self.runner = component.buffer
        self.running_infos = []
        assert self.args.eval_freq % self.args.steps_per_epoch == 0, \
            "ppo requires eval freq be the multiple of steps per epoch, otherwise it is never evaluated"
        self.populate(0)

    def training_step(self, batch, batch_idx, optimizer_idx):

        loss, train_info = self.agent.compute_loss(batch, optimizer_idx)
        if optimizer_idx == 0 and self.num_steps % self.args.log_freq == 0:
            for info_name, info_value in train_info.items():
                self.log("losses/{}".format(info_name), info_value, prog_bar=True)

        if optimizer_idx == 0 and self.num_steps % self.args.eval_freq == 0:
            self.evaluate(self.args.eval_episodes)
        return loss

    def on_epoch_end(self) -> None:
        prefix = 'train/'
        merged_info = merge_dicts(self.running_infos)
        for k, v in merged_info.items():
            self.log(prefix + k, v, prog_bar='epi_returns' in k)
        self.log(prefix + 'steps', self.num_steps, prog_bar=True)
        self.running_infos = []

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        optimizer_actor = torch.optim.Adam(self.agent.actor.parameters(), lr=self.args.actor_lr)
        optimizer_critic = torch.optim.Adam(self.agent.critic.parameters(), lr=self.args.critic_lr)

        return [optimizer_actor, optimizer_critic]

    # def optimizer_step(self, *args, **kwargs):
    #     for _ in range(self.args.nb_optim_iters):
    #         super().optimizer_step(*args, **kwargs)

    def get_dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = ExperienceSourceDataset(self.runner)
        dataloader = DataLoader(dataset=dataset, batch_size=self.args.batch_size)
        return dataloader

    def explore_schedule(self, num_steps):
        return 0

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lam", type=float, default=0.95, help="advantage discount factor")
        parser.add_argument("--actor_lr", type=float, default=3e-4, help="learning rate of actor network")
        parser.add_argument("--critic_lr", type=float, default=1e-3, help="learning rate of critic network")
        parser.add_argument("--max_episode_len", type=int, default=1000, help="capacity of the replay buffer")
        parser.add_argument("--eval_freq", type=int, default=8192, help="evaluation freq")
        parser.add_argument("--eval_episodes", type=int, default=5, help="Number of eval episodes")
        parser.add_argument("--hidden_size", type=int, default=128, help="Size of hidden layers")
        parser.add_argument("--log_freq", type=int, default=1024, help="logging freq")
        parser.add_argument("--ent_coef", type=float, default=0.01, help="coefficient for entropy bonus")
        parser.add_argument("--vf_coef", type=float, default=0.5, help="coefficient for entropy bonus")

        # parser.add_argument("--batch_size", type=int, default=512, help="batch_size when training network")
        #参考了openai baseline
        parser.add_argument(
            "--steps_per_epoch",
            type=int,
            default=2048,
            help="how many action-state pairs to rollout for trajectory collection per epoch"
        )
        parser.add_argument(
            "--nb_optim_iters", type=int, default=4, help="how many steps of gradient descent to perform on each batch"
        )
        parser.add_argument(
            "--clip_ratio", type=float, default=0.2, help="hyperparameter for clipping in the policy objective"
        )

        return parser
