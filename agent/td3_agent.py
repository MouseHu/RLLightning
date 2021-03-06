import torch
import torch.nn.functional as F
from agent.actor_critic_agent import ActorCriticAgent
from network.td3_model import Actor, Critic


class TD3Agent(ActorCriticAgent):
    def __init__(self, args, component) -> None:
        super(TD3Agent,self).__init__(args,component)
        state_dim = self.state_dim
        action_dim = self.action_dim
        max_action = self.max_action

        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.update_target()

    def compute_loss(self, batch) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        states, actions, rewards, dones, next_states = batch

        with torch.no_grad():
            noise = (
                    torch.randn_like(actions) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_actions = (
                    self.actor_target(next_states) + noise
            ).clamp(-self.max_action, self.max_action)

            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * torch.min(target_q1, target_q2)

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        actor_loss = -self.critic.Q1(states, self.actor(states)).mean()

        return critic_loss, actor_loss, torch.mean(q1), torch.mean(q1), torch.mean(target_q)
