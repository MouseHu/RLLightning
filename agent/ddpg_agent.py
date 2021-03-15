import torch
import torch.nn.functional as F
from agent.actor_critic_agent import ActorCriticAgent
from network.ddpg_model import Actor, Critic


class DDPGAgent(ActorCriticAgent):
    def __init__(self, args, component) -> None:
        super(DDPGAgent,self).__init__(args,component)
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

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_tp1 = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * q_tp1

        q = self.critic(states, actions)
        critic_loss = F.mse_loss(q, target_q)
        actor_loss = -self.critic(states, self.actor(states)).mean()

        train_info = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "q_mean": torch.mean(q),
            "q_target_mean": torch.mean(target_q)
        }

        if optimizer_idx == 0:
            loss = critic_loss
        else:
            loss = actor_loss

        return loss, train_info

