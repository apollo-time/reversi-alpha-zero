import os
from datetime import datetime
from logging import getLogger
from time import time
import copy
from random import random
from collections import namedtuple
import numpy as np
from lockfile.mkdirlockfile import MkdirLockFile
from reversi_zero.agent.player import SelfPlayer
from reversi_zero.config import Config
from reversi_zero.env.reversi_env import ReversiEnv
from reversi_zero.lib import tf_util
from reversi_zero.lib.data_helper import get_game_data_filenames, write_game_data_to_file

logger = getLogger(__name__)

ResignInfo = namedtuple("ResignInfo", "predict, actual")
ResignCtrl = namedtuple("ResignCtrl", "v, n, f_p_n")  # v_resign, total_n, false_positive_n

def start(config: Config, gpu_mem_frac=None):
    if gpu_mem_frac is not None:
        config.model.gpu_mem_frac = gpu_mem_frac

    return SelfPlayWorker(config, env=ReversiEnv()).start()


class SelfPlayWorker:
    def __init__(self, config: Config, env=None, model=None):
        """

        :param config:
        :param ReversiEnv|None env:
        :param reversi_zero.agent.model.ReversiModel|None model:
        """
        self.config = config
        self.model = model
        self.env = env

    def update_v_resign(self, resign_info, resign_ctrl):

        info = resign_info
        ctrl = resign_ctrl
        v_resign_delta = self.config.play.v_resign_delta
        fraction_t_max = self.config.play.v_resign_false_positive_fraction_t_max
        fraction_t_min = self.config.play.v_resign_false_positive_fraction_t_min
        min_n = self.config.play.v_resign_check_min_n

        v, n, f_p_n = ctrl.v, ctrl.n, ctrl.f_p_n

        n += 1
        f_p_n += 0 if info.predict == info.actual else 1

        fraction = float(f_p_n) / n
        logger.debug(f'resign f_p frac: {f_p_n} / {n} = {fraction}')
        if n >= min_n and fraction > fraction_t_max:
            v -= v_resign_delta
        elif n >= min_n and fraction < fraction_t_min:
            v += v_resign_delta
        else:
            pass

        if abs(ctrl.v - v) > 1e-10:
            logger.debug(f'#false_positive={f_p_n}, #n={n}, frac={fraction}, target_fract=[{fraction_t_min},{fraction_t_max}]')
            logger.debug(f'v_resign change from {ctrl.v} to {v}')
        return ResignCtrl(v, n, f_p_n)

    def start(self):
        if self.model is None:
            self.model = self.load_model()

        buffer = []
        game_idx = 1
        resign_ctrl = ResignCtrl(self.config.play.v_resign_init, 0, 0)

        logger.debug("game is on!!!")
        while True:
            start_time = time()

            prop = self.config.play.v_resign_disable_prop
            should_resign = random() >= prop
            moves, resign_info = self.play_a_game(should_resign, resign_ctrl.v)
            buffer += moves

            end_time = time()
            logger.debug(f"play game {game_idx} time={end_time - start_time} sec")

            if resign_info.predict is not None:
                resign_ctrl = self.update_v_resign(resign_info, resign_ctrl)

            if (game_idx % self.config.play_data.nb_game_in_file) == 0:
                self.save_play_data(buffer)
                buffer = []
                self.remove_old_play_data()

                # TODO: maybe this reload time is not consistent with the AZ paper...
                self.model.load(self.config.resource.model_dir)

            game_idx += 1

    def play_a_game(self, should_resign, v_resign):
        env = self.env.copy()
        env.reset()

        def make_sim_env_fn():
            return env.copy()

        player = SelfPlayer(make_sim_env_fn=make_sim_env_fn, config=self.config, model=self.model)
        player.prepare(root_env=env, dir_noise=True)

        moves = []
        resign_predicted_winner = None

        while not env.done:
            tau = 1 if env.turn < self.config.play.change_tau_turn else 0
            act, pi, vs = player.think(tau)

            if all(v < v_resign for v in vs):
                if should_resign:
                    logger.debug(f'Resign: v={vs[0]:.4f}, child_v={vs[1]:.4f}, thres={v_resign:.4f}')
                    env.resign()
                    break
                if resign_predicted_winner is None:
                    resign_predicted_winner = env.last_player

            moves += [[env.compress_ob(env.observation).tolist(), np.asarray(pi).tolist(), np.asarray(env.legal_moves).tolist()]]

            env.step(act)
            player.play(act)

        if env.black_wins:
            z = 1
        elif env.black_loses:
            z = -1
        else:
            z = 0
        for i, move in enumerate(moves):
            move += [z if i%2==0 else -z]

        resign_info = ResignInfo(resign_predicted_winner, env.winner)

        return moves, resign_info

    def save_play_data(self, buffer):
        rc = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        with MkdirLockFile(rc.play_data_dir):
            path = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % game_id)
            logger.info(f"save play data to {path}")
            write_game_data_to_file(path, buffer)

    def remove_old_play_data(self):
        with MkdirLockFile(self.config.resource.play_data_dir):
            files = get_game_data_filenames(self.config.resource)
            if len(files) < self.config.play_data.max_file_num:
                return
            for i in range(len(files) - self.config.play_data.max_file_num):
                os.remove(files[i])

    def load_model(self):
        from reversi_zero.agent.model import ReversiModel
        model = ReversiModel(self.config)
        rc = self.config.resource
        model.create_session()
        model.load(rc.model_dir)
        return model
