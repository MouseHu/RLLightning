from algorithm.dqn import DQNLearner
from algorithm.actor_critic import TD3Learner
from algorithm.actor_critic import SACLearner
from algorithm.actor_critic import DDPGLearner
from algorithm.ppo import PPOLearner
algo_list = {
    'dqn': DQNLearner,
    'td3': TD3Learner,
    'sac': SACLearner,
    'ddpg': DDPGLearner,
    'ppo': PPOLearner
}
