import enum
from logging import getLogger

from reversi_zero.agent.player import EvaluatePlayer
from reversi_zero.config import Config
from reversi_zero.env.reversi_env import Player, ReversiEnv
from reversi_zero.lib.bitboard import find_correct_moves
import numpy as np
logger = getLogger(__name__)

GameEvent = enum.Enum("GameEvent", "update ai_move over pass")


class PlayWithHuman:
    def __init__(self, config: Config, model_dir):
        self.config = config
        self.human_color = None
        self.observers = []
        self.env = ReversiEnv().reset()
        self.model = self._load_model(model_dir)
        self.ai = None  # type: EvaluatePlayer
        self.ai_confidence = None

    def add_observer(self, observer_func):
        self.observers.append(observer_func)

    def notify_all(self, event):
        for ob_func in self.observers:
            ob_func(event)

    def start_game(self, human_is_black):
        self.human_color = Player.black if human_is_black else Player.white
        self.env = ReversiEnv().reset()
        def make_sim_env_fn():
            return self.env.copy()
        self.ai = EvaluatePlayer(make_sim_env_fn=make_sim_env_fn, config=self.config, model=self.model)
        self.ai.prepare(self.env, dir_noise=False)
        self.ai_confidence = None

    def play_next_turn(self):
        self.notify_all(GameEvent.update)

        if self.over:
            self.notify_all(GameEvent.over)
            return

        if self.next_player != self.human_color:
            self.notify_all(GameEvent.ai_move)
        elif np.amax(self.env.legal_moves) == 0:
            # pass
            print('pass move')
            pos = 64
            self.env.step(pos)
            self.ai.play(pos, self.env)

    @property
    def over(self):
        return self.env.done

    @property
    def next_player(self):
        return self.env.next_player

    def stone(self, px, py):
        """left top=(0, 0), right bottom=(7,7)"""

        pos = int(py * 8 + px)
        assert 0 <= pos < 64
        bit = 1 << pos
        if self.env.board.black & bit:
            return Player.black
        elif self.env.board.white & bit:
            return Player.white
        return None

    @property
    def number_of_black_and_white(self):
        return self.env.board.number_of_black_and_white

    def available(self, px, py):
        pos = int(py * 8 + px)
        if pos < 0 or 64 <= pos:
            return False
        own, enemy = self.env.board.black, self.env.board.white
        if self.human_color == Player.white:
            own, enemy = enemy, own
        legal_moves = find_correct_moves(own, enemy)
        return legal_moves & (1 << pos)

    def move(self, px, py):
        pos = int(py * 8 + px)
        assert 0 <= pos < 64

        if self.next_player != self.human_color:
            raise Exception('not human\'s turn!')

        self.env.step(pos)

        self.ai.play(pos, self.env)

    def _load_model(self, model_dir):
        from reversi_zero.agent.model import ReversiModel
        model = ReversiModel(self.config)
        model.create_session()
        model.load(model_dir)

        return model

    def move_by_ai(self):
        if self.next_player == self.human_color:
            raise Exception('not AI\'s turn!')

        logger.info('start thinking...')
        action, _, vs = self.ai.think()
        self.ai_confidence = vs
        logger.info('end thinking...')
        self.env.step(action)
        self.ai.play(action, self.env)

    def get_state_of_next_player(self):
        if self.next_player == Player.black:
            own, enemy = self.env.board.black, self.env.board.white
        else:
            own, enemy = self.env.board.white, self.env.board.black
        return own, enemy


