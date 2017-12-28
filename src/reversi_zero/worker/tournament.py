import os
import copy
import json
from logging import getLogger
from os.path import join as ospj
from random import random
from time import sleep
from collections import namedtuple
import itertools


from reversi_zero.agent.model import ReversiModel
from reversi_zero.agent.player import EvaluatePlayer
from reversi_zero.config import Config
from reversi_zero.env.reversi_env import ReversiEnv, Player
from reversi_zero.lib import tf_util
from reversi_zero.lib.data_helper import get_next_generation_model_dirs
from reversi_zero.lib.model_helpler import save_as_best_model, load_best_model_weight, back_up_current_best_model

logger = getLogger(__name__)

TournamentPlayer = namedtuple('TournamentPlayer', ['name', 'config', 'weight'])
PlayerPlayerResult = namedtuple('PlayerPlayerResult', ['p1', 'p2', 'result'])


def start(config: Config, gpu_mem_frac=None):
    if gpu_mem_frac is not None:
        tf_util.set_session_config(per_process_gpu_memory_fraction=gpu_mem_frac)
    return TournamentWorker(config).start()


class TournamentWorker:
    """
    Just for fun: every model competes with all other models and see who is the champion :)
    Rule: every 2 models plays 9 games( draw games not counted ), the winner takes 1 point.
    """
    def __init__(self, config: Config):
        """

        :param config:
        """
        self.config = config
        dir = self.config.resource.model_dir
        self.TOURNAMENT_RESULT_PATH = ospj(dir, 'tournament_result.json')
        self.HASH_SPLIT = '#'
        self.players = self.register_players()

    def register_players(self):
        rc = self.config.resource
        dir = rc.model_dir

        # collect all candidates
        players = [
            TournamentPlayer('best', ospj(dir, rc.model_best_config_filename), ospj(dir, rc.model_best_weight_filename))
        ]
        files = os.listdir(dir)
        for f in files:
            prefix = f'{rc.model_best_weight_filename}.'
            if not f.startswith(prefix):
                continue
            v = f[len(prefix):]
            weight_path = os.path.join(dir, f)
            config_name = f'{rc.model_best_config_filename}.{v}'
            if config_name not in files:
                continue
            config_path = os.path.join(dir, config_name)
            players.append(TournamentPlayer(v, config_path, weight_path))

        return players

    def start(self):
        # hash players
        hasher = ReversiModel(self.config)  # only for hash usage
        hash_to_player_dict = dict()
        for p in self.players:
            hash = hasher.fetch_digest(p.weight)
            if hash in hash_to_player_dict:
                raise Exception(f'{p.name} is same with {hash_to_player_dict[hash].name}')
            hash_to_player_dict[hash] = p

        if len(hash_to_player_dict) < 2:
            raise Exception(f'only {len(hash_to_player_dict)} players to compete!')

        hashes_to_result_dict = self.load_hashes_to_result_dict()

        # tournament settings
        n_games = 1  # FIXME
        assert n_games % 2 == 1  # in case you change n_games value
        ignore_draws = True

        # now let's play!
        for hash1, hash2 in itertools.combinations(hash_to_player_dict, 2):

            result = []

            if hash1 > hash2:  # string comparasion is fine
                hash1, hash2 = hash2, hash1

            hashes = self.HASH_SPLIT.join(([hash1, hash2]))
            if hashes in hashes_to_result_dict:
                try:
                    result = list(hash_to_result_dict[hashes])
                except Exception as e:
                    logger.warning(e)
                    pass

            if len(result) >= n_games:
                # no chance to re-play though :P
                continue

            p1 = hash_to_player_dict[hash1]
            p2 = hash_to_player_dict[hash2]
            model1 = self.load_model(p1.config, p1.weight)
            model2 = self.load_model(p2.config, p2.weight)

            n_left_games = n_games - len(result)
            logger.info(f'{p1.name} and {p2.name} are playing {n_left_games} games...')
            new_result = self.play_n_games(model1, model2, n_left_games, ignore_draws)
            result.extend(new_result)
            assert len(result) == n_games

            hashes_to_result_dict[hashes] = result
            self.save_hashes_to_result_dict(hashes_to_result_dict)  # save in time in case future crashes

        # print result
        for hash, result in hashes_to_result_dict.items():
            hashes = hash.split(self.HASH_SPLIT)
            name1 = hash_to_player_dict[hashes[0]].name
            name2 = hash_to_player_dict[hashes[1]].name
            logger.info(f'{name1} v.s. {name2} : {result}')

    def load_hashes_to_result_dict(self):
        if os.path.exists(self.TOURNAMENT_RESULT_PATH):
            with open(self.TOURNAMENT_RESULT_PATH, "rt") as f:
                return json.load(f)
        else:
            return dict()

    def save_hashes_to_result_dict(self, the_dict):
        with open(self.TOURNAMENT_RESULT_PATH, "wt") as f:
            json.dump(the_dict, f)

    def load_model(self, config_path, weight_path):
        model = ReversiModel(self.config)
        model.load(config_path, weight_path)
        return model

    def play_n_games(self, model1, model2, n_games, ignore_draws):
        result = []
        for i in range(n_games):
            logger.info(f'game {i} is playing...')
            p1_win = self.play_game(model1, model2)
            while ignore_draws and p1_win is None:
                logger.info('draw. Replay...')
                p1_win = self.play_game(model1, model2)

            if p1_win is None:  # draw
                continue

            if p1_win:
                result.append(1)
            else:
                result.append(0)

        return result

    def play_game(self, model_1, model_2):
        env = ReversiEnv().reset()

        def make_sim_env_fn():
            return env.copy()

        p1 = EvaluatePlayer(make_sim_env_fn=make_sim_env_fn, config=self.config,
                            model=model_1, play_config=self.config.eval.play_config)
        p1.prepare(env, dir_noise=False)

        p2 = EvaluatePlayer(make_sim_env_fn=make_sim_env_fn, config=self.config,
                            model=model_2, play_config=self.config.eval.play_config)
        p2.prepare(env, dir_noise=False)

        p1_is_black = random() < 0.5
        if p1_is_black:
            black, white = p1, p2
        else:
            black, white = p2, p1

        while not env.done:
            if env.next_player == Player.black:
                action, _, _ = black.think()
            else:
                action, _, _ = white.think()

            env.step(action)

            black.play(action, env)
            white.play(action, env)

        if env.black_wins:
            p1_win = p1_is_black
        elif env.black_loses:
            p1_win = not p1_is_black
        else:
            p1_win = None

        return p1_win
