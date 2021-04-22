import argparse
import json
import os
import time
from argparse import Namespace

import pytorch_lightning as pl

from agent import agent_list
from algorithm import algo_list
from buffer import buffer_list
from utils.create_env import *


def get_meta_args():
    # get basic arguments
    parser = argparse.ArgumentParser(description='RL Meta Argparser')
    parser.add_argument('--agent', help='backend agent', type=str, default='dqn')
    parser.add_argument('--algo', help='backend algorithm', type=str, default='dqn')
    #FIXME:algo和agent可以不一致嘛？能不能只用一个参数
    parser.add_argument("--env_type", help='which type of env to use', type=str, default='toy')
    parser.add_argument("--eval_on_same", help='whether or not to eval on same env', type=bool, default=True)
    parser.add_argument('--load_json',
                        help='Load settings from file in json format. Command line options override values in file.')
    args, _ = parser.parse_known_args()
    
    #print(args) correct
    return args


def get_parser(meta_args):
    parser = argparse.ArgumentParser(description='RL Argparser')

    # basic
    parser.add_argument("--seed", type=int, default=int(time.time()), dest="seed")
    parser.add_argument("--comment", type=str, default="test")
    parser.add_argument('--gamma', help='discount factor', type=np.float32, default=0.99)
    parser.add_argument("--total_steps", type=int, default=int(2e6))
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")

    # environment
    parser = add_env_args(parser, meta_args)

    # algorithm
    lit_model = algo_list.get(meta_args.algo)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = lit_model.add_model_specific_args(parser)

    # buffer
    parser.add_argument('--buffer', help='type of replay buffer', type=str, default='default')
    parser.add_argument('--batch_size', help='size of sample batch', type=int, default=256)
    parser.add_argument('--buffer_size', help='number of transitions in replay buffer', type=np.int32, default=50000)
    parser.add_argument('--warmup', help='number of timesteps for buffer warmup', type=np.int32, default=10000)

    return parser


def get_args():
    # get all arguments
    meta_args = get_meta_args()
    args = Namespace()
    if meta_args.load_json:
        with open(meta_args.load_json, 'rt') as f:
            json_dict = json.load(f)
            meta_json_dict = {k: v for k, v in json_dict.items() if k in meta_args.__dict__.keys()}
            args.__dict__.update(json_dict)
            meta_args.__dict__.update(meta_json_dict)
    args, _ = get_parser(meta_args).parse_known_args(namespace=args)
    args.__dict__.update(meta_args.__dict__)
    return args


def get_path(args):
    # get log directory
    base_path = os.getenv('RL_LOGDIR', "/home/lzy/myRL")
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
