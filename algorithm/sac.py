from argparse import Namespace

import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from typing import Tuple, List

from algorithm.actor_critic import ActorCriticLearner


class SACLearner(ActorCriticLearner):
    def __init__(self, args: Namespace, components: Namespace) -> None:
        super().__init__(args, components)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], nb_batch, optimizer_idx):
        if self.global_step % self.args.training_steps == 0:
            self.rollout(num_step=self.args.training_freq)
        # Calculates training loss
        q_loss, value_loss, actor_loss, ent_coef_loss, q1, q2, v, target_q, v_backup = self.agent.compute_loss(batch)

        # Soft update of target network
        if self.global_step % self.args.target_freq == 0:
            self.agent.update_target(tau=self.args.tau)

        # Evaluation
        if self.num_steps % self.args.eval_freq == 0:
            self.rollout(num_step=self.args.max_test_step, train=False)

        self.log("losses/q_loss", q_loss)
        self.log("losses/value_loss", value_loss)
        self.log("losses/actor_loss", actor_loss)
        self.log("losses/q1_mean", q1)
        self.log("losses/q2_mean", q2)
        self.log("losses/q_target_mean", target_q)
        self.log("losses/v_target_mean", v_backup)

        if optimizer_idx == 0:
            return q_loss+value_loss
        elif optimizer_idx == 1 and self.num_steps % self.args.policy_delay == 0:
            return actor_loss
        elif optimizer_idx == 2:
            return ent_coef_loss
        return 0

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
