from algorithm.dqn import DQNLearner
from algorithm.td3 import TD3Learner
from algorithm.sac import SACLearner
from algorithm.ddpg import DDPGLearner

algo_list = {
    'dqn': DQNLearner,
    'td3': TD3Learner,
    'sac': SACLearner,
    'ddpg': DDPGLearner
}
