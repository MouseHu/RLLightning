import argparse
import os
import time
from argparse import Namespace

import pytorch_lightning as pl

from agent import agent_list
from algorithm import algo_list
from buffer import buffer_list
from utils.create_env import *


def get_args():
    parser = argparse.ArgumentParser(description='RL Argparser')

    # basic
    parser.add_argument("--seed", type=int, default=int(time.time()), dest="seed")
    parser.add_argument("--comment", type=str, default="test")
    parser.add_argument('--gamma', help='discount factor', type=np.float32, default=0.99)
    parser.add_argument("--total_steps", type=int, default=int(2e6))
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument('--agent', help='backend agent', type=str, default='dqn')
    parser.add_argument('--algo', help='backend algorithm', type=str, default='dqn')
    parser.add_argument("--env_type", help='which type of env to use', type=str, default='Atari')
    args, _ = parser.parse_known_args()

    # environment
    parser.add_argument("--eval_on_same", help='whether or not to eval on same env', type=bool, default=True)
    parser = add_env_args(parser, args)

    # algorithm
    lit_model = algo_list.get(args.algo)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = lit_model.add_model_specific_args(parser)

    # buffer
    parser.add_argument('--buffer', help='type of replay buffer', type=str, default='default')
    parser.add_argument('--batch_size', help='size of sample batch', type=np.int32, default=32)
    parser.add_argument('--buffer_size', help='number of transitions in replay buffer', type=np.int32, default=200000)
    parser.add_argument('--warmup', help='number of timesteps for buffer warmup', type=np.int32, default=10000)

    args = parser.parse_args()
    return args


def get_path(args):
    base_path = os.getenv('RL_LOGDIR', "/data1/hh/myRL")
    cur_time = time.strftime('%m%d_%H:%M:%S')
    path = os.path.join(base_path, "{}_{}_{}".format(args.env_name, args.comment, cur_time))
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def setup(args):
    component = Namespace()

    env = component.env = create_env(args)
    eval_env = component.eval_env = create_env(args, eval=not args.eval_on_same)
    buffer = component.buffer = buffer_list[args.buffer](args, component)
    agent = component.agent = agent_list[args.agent](args, component)
    learner = component.learner = algo_list[args.algo](args, component)

    return env, eval_env, buffer, agent, learner
