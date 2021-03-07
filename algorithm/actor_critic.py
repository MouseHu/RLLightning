import argparse
from argparse import Namespace

import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from typing import Tuple, List

from algorithm.base_learner import BaseLearner


class ActorCriticLearner(BaseLearner):
    def __init__(self, args: Namespace, components: Namespace) -> None:
        super().__init__(args, components)
        self.hparams = args
        print("Warming up ...")
        self.populate(self.args.warmup)

    def explore_schedule(self, num_steps):
        return 0

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], nb_batch, optimizer_idx):
        # Calculate training loss
        loss, train_info = self.agent.compute_loss(batch)

        if optimizer_idx == 0:
            if self.global_step % self.args.training_steps == 0:
                self.rollout(num_step=self.args.training_freq)
            # Soft update of target network
            if self.global_step % self.args.target_freq == 0:
                self.agent.update_target(tau=self.args.tau)

            # Evaluation
            if self.global_step % self.args.eval_freq == 0:
                self.evaluate(num_episode=self.args.eval_episodes)

            if self.global_step % self.args.log_freq == 0:
                for info_name, info_value in train_info.items():
                    self.log("losses/{}".format(info_name), info_value)

        return loss

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer"""
        critic_optimizer = optim.Adam(self.agent.critic.parameters(), lr=self.args.critic_lr)
        actor_optimizer = optim.Adam(self.agent.actor.parameters(), lr=self.args.actor_lr)
        return [critic_optimizer, actor_optimizer]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--target_freq", type=int, default=100,
                            help="how many frames do we update the target network")
        parser.add_argument("--training_freq", type=int, default=100, help="how frequently do we train our network")
        parser.add_argument("--training_steps", type=int, default=100,
                            help="how many step do we train our network")
        parser.add_argument("--max_test_step", type=int, default=5000, help="max steps for testing")
        parser.add_argument("--policy_delay", type=int, default=2, help="delay policy update")
        parser.add_argument("--eval_freq", type=int, default=5000, help="evaluation freq")
        parser.add_argument("--log_freq", type=int, default=100, help="loggging freq")
        parser.add_argument("--tau", type=float, default=0.005, help="Polyak averaging coefficient")
        parser.add_argument("--policy_noise", type=float, default=0.2, help="Exloration std for gaussian noise")
        parser.add_argument("--actor_lr", type=float, default=3e-4, help="Learning rate for actor")
        parser.add_argument("--critic_lr", type=float, default=3e-4, help="Learning rate for critic")
        parser.add_argument("--explore_noise", type=float, default=0.1, help="Exloration std for gaussian noise")
        parser.add_argument("--eval_episodes", type=int, default=5, help="Number of eval episodes")

        parser.add_argument("--noise_clip", type=float, default=0.5,
                            help="Action noise clip for target policy smoothing")
        return parser


class DDPGLearner(ActorCriticLearner):
    def __init__(self, args: Namespace, components: Namespace) -> None:
        super().__init__(args, components)


class TD3Learner(ActorCriticLearner):
    def __init__(self, args: Namespace, components: Namespace) -> None:
        super().__init__(args, components)


class SACLearner(ActorCriticLearner):
    def __init__(self, args: Namespace, components: Namespace) -> None:
        super().__init__(args, components)

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer"""
        critic_optimizer = optim.Adam(self.agent.actor.parameters(), lr=self.args.critic_lr)
        actor_optimizer = optim.Adam(self.agent.critic.parameters(), lr=self.args.actor_lr)
        optimizers = [critic_optimizer, actor_optimizer]
        if 'auto' in self.args.ent_coef:
            entropy_optimizer = optim.Adam(self.ent_coef, lr=self.args.entropy_lr)
            optimizers.append(entropy_optimizer)
        return optimizers

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--target_entropy", type=str, default="auto", help="target entropy")
        parser.add_argument("--ent_coef", type=str, default="auto", help="ent_coef")

        return parser
