import torch
import torch.nn.functional as F

from agent.actor_critic_agent import ActorCriticAgent
from network.td3_model import Actor, Critic


class TD3Agent(ActorCriticAgent):
    def __init__(self, args, component) -> None:
        super(TD3Agent, self).__init__(args, component)
        state_dim = self.state_dim
        action_dim = self.action_dim
        max_action = self.max_action

        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.update_target()

    def compute_loss(self, batch, optimizer_idx) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        states, actions, rewards, dones, next_states = batch
        # print(states,actions)
        with torch.no_grad():
            noise = (
                    torch.randn_like(actions) * self.args.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_actions = (
                    self.actor_target(next_states) + noise
            ).clamp(-self.max_action, self.max_action)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * torch.min(target_q1, target_q2)
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        actor_loss = -self.critic.Q1(states, self.actor(states)).mean()

        train_info = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "q1_mean": torch.mean(q1),
            "q2_mean": torch.mean(q2),
            "target_q": torch.mean(target_q)
        }

        if optimizer_idx == 0:
            loss = critic_loss
        elif optimizer_idx == 1 and self.component.learner.num_steps % self.args.policy_delay == 0:
            loss = actor_loss
        else:
            loss = 0
        return loss, train_info
