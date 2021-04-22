from itertools import chain

import gym
import torch
import torch.nn as nn
from gym.spaces import Box, Discrete

from agent.base_agent import Agent
from network.basic_model import MLP,NatureCNN
from network.ppo_model import ActorContinous, ActorCategorical
from agent.actor_critic_agent import ActorCriticAgent
from agent.a2c_agent import A2CAgent


class PPOAgent(A2CAgent):
    """
    Implementation of Policy Gradient Based Agent
    """

    def __init__(self, args, component) -> None:
        A2CAgent.__init__(self, args, component)
        self.clip_ratio=args.clip_ratio

    def compute_loss(self, batch, optimizer_idx) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        state, action, old_logp, qval, adv = batch
        action, old_logp, adv = action.squeeze(), old_logp.squeeze(), adv.squeeze()
        adv = (adv - adv.mean()) / (adv.std()+1e-8)
        # TODO: Refer to Baselines for a better implementation

        pi, _ = self.actor(state)
        logp = self.actor.get_log_prob(pi, action)
        ratio = torch.exp(logp - old_logp)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        actor_loss = -(torch.min(ratio * adv, clip_adv)).mean()

        value = self.critic(state)
        critic_loss = self.args.vf_coef * (qval - value).pow(2).mean()
        entropy_loss = (- self.args.ent_coef * self.actor.entropy(pi)).mean()

        train_info = {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy_loss": entropy_loss
        }

        if optimizer_idx == 0:
            loss = actor_loss + entropy_loss
        elif optimizer_idx == 1:
            loss = critic_loss
        else:
            loss = None
        return loss, train_info