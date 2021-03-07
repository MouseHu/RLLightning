import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.actor_critic_agent import ActorCriticAgent
from network.sac_model import Actor, Critic


class SACAgent(ActorCriticAgent):
    def __init__(self, args, component) -> None:
        super(SACAgent, self).__init__(args, component)
        state_dim = self.state_dim
        action_dim = self.action_dim
        max_action = self.max_action

        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.update_target()

        self.setup_entropy()

    def policy(self, state):
        deterministic_policy, policy, _, _ = self.actor(state)
        return deterministic_policy, policy

    def get_action(self, state, epsilon: float, train=True) -> int:
        if np.random.random() < epsilon:
            env = self.env if train else self.eval_env
            action = env.action_space.sample()
        else:
            deterministic_action, stochastic_action = self.policy(state)
            action = stochastic_action if train else deterministic_action
            action = action.cpu().data.numpy().flatten()
        return action

    def compute_loss(self, batch, optimizer_idx) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        states, actions, rewards, dones, next_states = batch

        _, actions_pi, logp_pi, entropy = self.actor(states)
        q1_pi, q2_pi, v = self.critic(states, actions_pi)
        q1, q2, v = self.critic(states, actions)

        with torch.no_grad():
            target_value = self.critic_target.Value(next_states)
            min_q_pi = torch.min(q1_pi, q2_pi)
            target_q = rewards + (1 - dones) * self.gamma * target_value
            v_backup = min_q_pi - self.ent_coef * logp_pi

        q_loss = 0.5 * (F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q))
        value_loss = 0.5 * F.mse_loss(v, v_backup)
        actor_loss = (self.ent_coef * logp_pi - q1).mean()

        if 'auto' in self.args.ent_coef:
            ent_coef_loss = -(self.log_ent_coef * (logp_pi + self.target_entropy).detach()).mean()
        else:
            ent_coef_loss = 0

        if optimizer_idx == 0:
            loss = q_loss + value_loss
        elif optimizer_idx == 1 and self.component.learner.num_steps % self.args.policy_delay == 0:
            loss = actor_loss
        elif optimizer_idx == 2:
            loss = ent_coef_loss
        else:
            loss = 0

        train_info = {
            "q_loss": q_loss,
            "value_loss": value_loss,
            "actor_loss": actor_loss,
            "entropy_loss": ent_coef_loss,
            "q1_mean": q1.mean(),
            "q2_mean": q2.mean(),
            "value_mean": v.mean(),
            "target_q_mean": target_q.mean(),
            "target_value_mean": v_backup.mean()
        }
        return loss, train_info

    def setup_entropy(self):

        if self.args.target_entropy == 'auto':
            self.target_entropy = -np.prod(self.action_space.shape).astype(np.float32)
        else:
            self.target_entropy = float(self.args.target_entropy)

        if isinstance(self.args.ent_coef, str) and self.args.ent_coef.startswith('auto'):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if '_' in self.args.ent_coef:
                init_value = float(self.args.ent_coef.split('_')[1])
                assert init_value > 0., "The initial value of ent_coef must be greater than 0"

            self.log_ent_coef = nn.Parameter(torch.tensor([init_value], requires_grad=True))
            self.ent_coef = self.log_ent_coef.exp()
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef = float(self.ent_coef)
