from itertools import chain

import gym
import torch
import torch.nn as nn
from gym.spaces import Box, Discrete

from agent.base_agent import Agent
from network.basic_model import MLP
from network.ppo_model import ActorContinous, ActorCategorical


class PPOAgent(Agent, nn.Module):
    """
    Implementation of Policy Gradient Based Agent
    """

    def __init__(self, args, component) -> None:
        Agent.__init__(self, component.env, component.eval_env, component.buffer, args)
        nn.Module.__init__(self)
        self.gamma = args.gamma
        self.clip_ratio = args.clip_ratio
        self.state_dim = self.env.observation_space.shape[0]

        hidden_size = self.args.hidden_size

        # value network
        self.critic = MLP([self.state_dim, hidden_size, hidden_size, 1])
        # policy network (agent)
        if isinstance(self.env.action_space, Box):
            self.act_dim = self.env.action_space.shape[0]
            self.actor = ActorContinous(MLP([self.state_dim, hidden_size, hidden_size, self.act_dim]), self.act_dim)
        elif isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
            self.act_dim = self.env.action_space.n
            self.actor = ActorCategorical(MLP([self.state_dim, hidden_size, hidden_size, self.act_dim]))
        else:
            raise NotImplementedError(
                'Env action space should be of type Box (continous) or Discrete (categorical).'
                f' Got type: {type(self.env.action_space)}'
            )

    def policy(self, state):
        return self.actor(state)

    def forward(self, state):
        return self.actor(state), self.critic(state)

    def parameters(self):
        return chain(self.actor.parameters(), self.critic.parameters())

    def get_action(self, state, epsilon: float, train=True):
        pi, action = self.actor(state)
        value = self.critic(state)
        log_prob = self.actor.get_log_prob(pi, action)

        action = action.cpu().numpy()
        log_prob = log_prob.cpu().numpy()
        value = value.cpu().numpy()

        if isinstance(self.env.action_space, Discrete):
            action = action.item()
        return log_prob, action, value

    def compute_loss(self, batch, optimizer_idx) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        state, action, old_logp, qval, adv = batch
        action, old_logp, adv = action.squeeze(), old_logp.squeeze(), adv.squeeze()
        adv = (adv - adv.mean()) / adv.std()
        # TODO: Refer to Baselines for a better implementation

        pi, _ = self.actor(state)
        logp = self.actor.get_log_prob(pi, action)
        ratio = torch.exp(logp - old_logp)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        actor_loss = -(torch.min(ratio * adv, clip_adv)).mean()

        value = self.critic(state)
        critic_loss = self.args.vf_coef * (qval - value).pow(2).mean()
        entropy_loss = - self.args.ent_coef * self.actor.entropy(pi)

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

    @torch.no_grad()
    def step(self, state, epsilon: float = 0.0, train=True):

        log_prob, action, value = self.get_action(state, epsilon, train)

        # do step in the environment
        env = self.env if train else self.eval_env
        new_state, reward, done, info = env.step(action)
        if train:
            self.state = new_state
        else:
            self.eval_state = new_state

        if done:
            self.reset(train)

        return new_state, reward, done, info, action, log_prob, value


class A2CAgent(PPOAgent):
    """
    Implementation of Policy Gradient Based Agent
    """

    # TODO: Conceptually, PPO Agent should inherit A2C Agent, rather than the other way around
    def __init__(self, args, component) -> None:
        super().__init__(args, component)

    def compute_loss(self, batch, optimizer_idx) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        state, action, old_logp, qval, adv = batch
        action, old_logp, adv = action.squeeze(), old_logp.squeeze(), adv.squeeze()
        value = self.critic(state)
        pi, _ = self.actor(state)
        logp = self.actor.get_log_prob(pi, action)

        actor_loss = -(adv * logp).mean()
        critic_loss = (qval - value).pow(2).mean()
        entropy_loss = - self.args.ent_coef * self.actor.entropy(pi)

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
