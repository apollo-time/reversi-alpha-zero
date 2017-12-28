import argparse

from logging import getLogger

from .lib.logger import setup_logger
from .config import Config

logger = getLogger(__name__)

CMD_LIST = ['self', 'opt', 'play_gui', 'tournament']


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", help="what to do", choices=CMD_LIST)
    parser.add_argument("--new", help="run from new best model", action="store_true")
    parser.add_argument("--total-step", help="set TrainerConfig.start_total_steps", type=int)
    parser.add_argument("--gpu_mem_frac", help="gpu memory fraction", default=None)
    return parser


def setup(config: Config, args, setup_logger_flag):
    config.opts.new = args.new
    if args.total_step is not None:
        config.trainer.start_total_steps = args.total_step
    config.resource.create_directories()
    if setup_logger_flag:
        setup_logger(config.resource.main_log_path)


def start():
    parser = create_parser()
    args = parser.parse_args()
    gpu_mem_frac = None
    if args.gpu_mem_frac:
        try:
            gpu_mem_frac = float(args.gpu_mem_frac)
        except ValueError:
            pass

    config = Config()
    setup_logger_flag = args.cmd != 'play_gui'
    setup(config, args, setup_logger_flag=setup_logger_flag)

    if args.cmd == "self":
        from .worker import self_play
        return self_play.start(config, gpu_mem_frac)
    elif args.cmd == 'opt':
        from .worker import optimize
        return optimize.start(config, gpu_mem_frac)
    elif args.cmd == 'play_gui':
        from .play_game import gui
        return gui.start(config)
    elif args.cmd == 'tournament':
        from .worker import tournament
        return tournament.start(config)
