import numpy as np
from typing import List

from utils.func_utils import array_min2d


class Runner(object):
    def __init__(self, args, components) -> None:
        self.args = args
        self.components = components

        self.gamma = args.gamma
        self.lam = args.lam
        self.steps_per_epoch = args.steps_per_epoch
        self.max_episode_len = args.max_episode_len
        self.agent = None
        self.learner = None

    def register(self):
        self.agent = self.components.agent
        self.learner = self.components.learner

    def generate_samples(self):
        if self.agent is None or self.learner is None:
            self.register()
        batch_states, batch_actions, batch_dones = [], [], []
        batch_infos, batch_qvals, batch_logp, batch_adv = [], [], [], []
        ep_rewards, ep_values = [], []
        #print(self.steps_per_epoch)256
        for step in range(self.steps_per_epoch):
            self.learner.num_steps += 1
            batch_states.append(self.agent.state)

            new_state, reward, done, info, action, log_prob, value = self.agent.step()

            batch_infos.append(info)
            batch_actions.append(action)
            batch_logp.append(log_prob)
            batch_dones.append(done)
            ep_rewards.append(reward)
            ep_values.append(value)

            epoch_end = step == (self.steps_per_epoch - 1)
            terminal = len(ep_rewards) == self.max_episode_len
            if epoch_end or done or terminal:
                if (terminal or epoch_end) and not done:
                    last_value = self.agent.critic(self.learner.get_state()).item()
                else:
                    last_value = 0
                # discounted cumulative reward
                batch_qvals += self.discount_rewards(ep_rewards + [last_value], self.gamma)[:-1]
                # advantage
                batch_adv += self.calc_advantage(ep_rewards, ep_values, last_value)
                ep_rewards = []
                ep_values = []
        self.learner.running_infos += batch_infos
        train_data = zip(
            array_min2d(batch_states), array_min2d(batch_actions), array_min2d(batch_logp),
            array_min2d(batch_qvals), array_min2d(batch_adv)
        )
        for state, action, logp_old, qval, adv in train_data:
            yield state, action, logp_old, qval, adv

    @staticmethod
    def discount_rewards(rewards: List[float], discount) -> List[float]:
        """Calculate the discounted rewards of all rewards in list
        Args:
            rewards: list of rewards/advantages
        Returns:
            list of discounted rewards/advantages
        """
        if isinstance(rewards[0], np.ndarray):
            rewards = [reward.item() for reward in rewards]
        assert isinstance(rewards[0], float)

        cumul_reward = []
        sum_r = 0.0

        for r in reversed(rewards):
            sum_r = (sum_r * discount) + r
            cumul_reward.append(sum_r)

        return list(reversed(cumul_reward))

    def calc_advantage(self, rewards: List[float], values: List[float], last_value: float) -> List[float]:
        """Calculate the advantage given rewards, state values, and the last value of episode
        Args:
            rewards: list of episode rewards
            values: list of state values from critic
            last_value: value of last state of episode
        Returns:
            list of advantages
        """
        if isinstance(rewards[0], np.ndarray):
            rewards = [reward.item() for reward in rewards]
        if isinstance(values[0], np.ndarray):
            values = [values.item() for values in values]
        rews = rewards + [last_value]
        vals = values + [last_value]

        # GAE
        delta = [rews[i] + self.gamma * vals[i + 1] - vals[i] for i in range(len(rews) - 1)]
        adv = self.discount_rewards(delta, self.gamma * self.lam)

        return adv
