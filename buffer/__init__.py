from buffer.nstep_replay_buffer import NStepReplayBuffer
from buffer.replay_buffer import ReplayBuffer
from buffer.stack_replay_buffer import StackReplayBuffer
from buffer.prioritized_replay_buffer import PrioritizedReplayBuffer
from buffer.runner import Runner

buffer_list = {
    "default": ReplayBuffer,
    "stacked": StackReplayBuffer,
    "nstep": NStepReplayBuffer,
    "prioritized": PrioritizedReplayBuffer,
    "runner": Runner
}
