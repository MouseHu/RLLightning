from agent.ddpg_agent import DDPGAgent
from agent.dqn_agent import DQNAgent
from agent.sac_agent import SACAgent
from agent.td3_agent import TD3Agent
from agent.ppo_agent import PPOAgent
from agent.ppo_agent import A2CAgent

agent_list = {
    "dqn": DQNAgent,
    "td3": TD3Agent,
    "sac": SACAgent,
    "ddpg": DDPGAgent,
    "ppo": PPOAgent,
    "a2c": A2CAgent,
    "dueling": DQNAgent,
    "double": DQNAgent,
    "noisy": DQNAgent
}
