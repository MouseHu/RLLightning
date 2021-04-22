from itertools import chain

import gym
import torch
import torch.nn as nn
from gym.spaces import Box, Discrete

from agent.base_agent import Agent
from network.basic_model import MLP,NatureCNN
from network.ppo_model import ActorContinous, ActorCategorical
from agent.actor_critic_agent import ActorCriticAgent

class A2CAgent(ActorCriticAgent):
    def __init__(self, args, component) -> None:
        ActorCriticAgent.__init__(self,args,component)
        hidden_size = self.args.hidden_size

        if isinstance(self.env.action_space, Box):
            raise NotImplementedError("not finished")
            self.act_dim = self.env.action_space.shape[0]
            self.actor = ActorContinous(MLP([self.state_dim, hidden_size,hidden_size,hidden_size, self.act_dim]), self.act_dim)
        
        elif isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
            self.act_dim = self.env.action_space.n
            obs_shape = self.env.observation_space.shape
            if len(obs_shape)==1:
                self.critic = MLP([self.state_dim, hidden_size, hidden_size,hidden_size,1])
                self.actor = ActorCategorical(MLP([self.state_dim, hidden_size, 
                    hidden_size, hidden_size ,hidden_size,self.act_dim]))
            else: #image input [4,84,84]
                self.critic = nn.Sequential(NatureCNN(obs_shape, hidden_size), \
                nn.Linear(hidden_size,1))
                self.actor = ActorCategorical(nn.Sequential(NatureCNN(obs_shape, hidden_size), \
                nn.Linear(hidden_size,self.act_dim)))  
            
        else:
            raise NotImplementedError(
                'Env action space should be of type Box (continous) or Discrete (categorical).'
                f' Got type: {type(self.env.action_space)}'
            )

    def compute_loss(self, batch, optimizer_idx) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        state, action, old_logp, qval, adv = batch

        action, old_logp, adv = action.squeeze(), old_logp.squeeze(), adv.squeeze()
        #normalize advantage
        adv = (adv - adv.mean()) / adv.std()

        value = self.critic(state)
        pi, _ = self.actor(state)
        logp = self.actor.get_log_prob(pi, action)
        actor_loss = -(adv.detach() * logp).mean()
        critic_loss = self.args.vf_coef * (qval.detach() - value).pow(2).mean()
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

    @torch.no_grad()
    def step(self, epsilon: float = 0.0, train=True):
        state = self.get_state(train = train)
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

    def get_action(self, state, epsilon: float, train=True):
        pi, action = self.actor(state)
        value = self.critic(state)
        log_prob = self.actor.get_log_prob(pi, action)#suppose discrete?

        action = action.cpu().numpy()
        log_prob = log_prob.cpu().numpy()
        value = value.cpu().numpy()

        if isinstance(self.env.action_space, Discrete):
            action = action.item()
        return log_prob, action, value
