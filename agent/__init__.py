from agent.ddpg_agent import DDPGAgent
from agent.dqn_agent import DQNAgent
from agent.sac_agent import SACAgent
from agent.td3_agent import TD3Agent

agent_list = {
    "dqn": DQNAgent,
    "td3": TD3Agent,
    "sac": SACAgent,
    "ddpg": DDPGAgent
}
