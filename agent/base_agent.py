import gym
import torch

from buffer.replay_buffer import ReplayBuffer


class Agent(object):
    def __init__(self, env: gym.Env, eval_env: gym.Env, replay_buffer: ReplayBuffer, args) -> None:
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.env = env
        self.eval_env = eval_env
        self.replay_buffer = replay_buffer
        self.args = args
        self.state = None
        self.eval_state = None
        self.reset()

    def policy(self, state):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

    def reset(self, train=True) -> None:
        """Resets the environment and updates the state"""
        if train:
            self.state = self.env.reset()
        else:
            if self.eval_env is not None:
                self.eval_state = self.eval_env.reset()

    def get_state(self, train=True):
        state = self.state if train else self.eval_state
        
        if len(state.shape) >= 1:  # image input
            state = state.astype(np.float32) / 255.0  
        state = torch.tensor([state], device=self.component.leaner.device, dtype=torch.float32)
        return state

    def get_action(self, state, epsilon: float, train=True):
        raise NotImplementedError

    def step(self, state, epsilon: float = 0.0, train=True):
        raise NotImplementedError