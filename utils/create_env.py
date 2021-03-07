from env.monitor import Monitor
from env.vanilla import VanillaEnv
from env.wrapper import *
from utils.os_utils import str2bool


def create_atari_env(args, eval=False):
    return VanillaEnv(args)


def create_mujoco_env(args, eval=False):
    env = gym.make(args.env_name if not eval else args.eval_env_name)
    env = TimestepWrapper(env)
    env = DelayedRewardWrapper(env, args.delay_step if not eval else args.eval_delay_step)
    env = Monitor(env, None)
    env = MonitorWrapper(env,args.gamma)
    return env


def create_toy_env(args, eval=False):
    return gym.make(args.env_name if not eval else args.eval_env_name)


def create_env(args, eval=False):
    if args.env_type in ["MuJoCo", "mujoco", "MUJOCO"]:
        return create_mujoco_env(args, eval)
    elif args.env_type in ["Atari", "atari"]:
        return create_atari_env(args, eval)
    elif args.env_type in ["fourrooms", "Fourrooms", "toy", "TOY"]:
        return create_toy_env(args, eval)
    else:
        raise NotImplementedError


def add_atari_args(parser):
    for prefix in ["", "eval_"]:
        parser.add_argument('--' + prefix + 'env_name', help='which atari env to use', type=str, default="CartPole-v0")
        parser.add_argument('--' + prefix + 'sticky', help='whether to use sticky actions', type=str2bool,
                            default=False)
        parser.add_argument('--' + prefix + 'noop', help='number of noop actions while starting new episode',
                            type=np.int32,
                            default=30)
        parser.add_argument('--' + prefix + 'frames', help='number of stacked frames', type=np.int32, default=4)
        parser.add_argument('--' + prefix + 'rews_scale', help='scale of rewards', type=np.float32, default=1.0)
        parser.add_argument('--' + prefix + 'test_eps', help='random action noise in atari testing', type=np.float32,
                            default=0.001)
    return parser


def add_mujoco_args(parser):
    for prefix in ["", "eval_"]:
        parser.add_argument('--' + prefix + 'env_name', help='which mujoco env to use', type=str,
                            default="HalfCheetah-v2")
        parser.add_argument('--' + prefix + 'delay_step', help='whether to use truly done signal', type=int, default=0)
        parser.add_argument('--' + prefix + 'truly_done', help='whether to use truly done signal', type=str2bool,
                            default=False)
    return parser


def add_toy_args(parser):
    # parser.add_argument('--env_name', help='which env to use', type=str, default="HalfCheetah-v2")
    return parser


def add_env_args(parser, args):
    if args.env_type in ["MuJoCo", "mujoco", "MUJOCO"]:
        parser = add_mujoco_args(parser)
    elif args.env_type in ["Atari", "atari"]:
        parser = add_atari_args(parser)
    elif args.env_type in ["fourrooms", "Fourrooms", "toy", "TOY"]:
        parser = add_toy_args(parser)
    else:
        raise NotImplementedError
    return parser
