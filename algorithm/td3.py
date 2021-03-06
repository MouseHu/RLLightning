from argparse import Namespace

import torch
from typing import Tuple

from algorithm.actor_critic import ActorCriticLearner


class TD3Learner(ActorCriticLearner):
    def __init__(self, args: Namespace, components: Namespace) -> None:
        super().__init__(args, components)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], nb_batch, optimizer_idx):
        if self.global_step % self.args.training_steps == 0:
            self.rollout(num_step=self.args.training_freq)
        # Calculates training loss
        critic_loss, actor_loss, q1_mean, q2_mean, q_target_mean = self.agent.compute_loss(batch)

        # Soft update of target network
        if self.global_step % self.args.target_freq == 0:
            self.agent.update_target(tau=self.args.tau)

        # Evaluation
        if self.num_steps % self.args.eval_freq == 0:
            self.rollout(num_step=self.args.max_test_step, train=False)

        self.log("losses/critic_loss", critic_loss)
        self.log("losses/actor_loss", actor_loss)
        self.log("losses/q1_mean", q1_mean)
        self.log("losses/q2_mean", q2_mean)
        self.log("losses/q_target_mean", q_target_mean)

        if optimizer_idx == 0:
            return critic_loss
        elif optimizer_idx == 1 and self.num_steps % self.args.policy_delay == 0:
            return actor_loss
        return 0
