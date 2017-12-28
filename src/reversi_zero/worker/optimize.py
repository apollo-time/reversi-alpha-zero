import os
from datetime import datetime
from logging import getLogger
from time import sleep
from random import randint
from bisect import bisect

import tensorflow as tf
import numpy as np
import time
from reversi_zero.env.reversi_env import ReversiEnv
from reversi_zero.config import Config
from reversi_zero.lib import tf_util
from reversi_zero.lib.data_helper import get_game_data_filenames, read_game_data_from_file
from lockfile.mkdirlockfile import MkdirLockFile

logger = getLogger(__name__)


def start(config: Config, gpu_mem_frac=None):
    if gpu_mem_frac is not None:
        config.model.gpu_mem_frac = gpu_mem_frac
    return OptimizeWorker(config).start()


class DataSet:
    def __init__(self, loaded_data):
        self.length_array = []
        self.filename_array = []

        total_length = 0
        for filename in loaded_data:
            length = len(loaded_data[filename])
            total_length += length

            self.length_array.append(total_length)
            self.filename_array.append(filename)

    def locate(self, index):
        p = bisect(self.length_array, index)
        offset = index - self.length_array[p-1] if p > 0 else index
        return self.filename_array[p], offset

    @property
    def size(self):
        return self.length_array[-1]


class OptimizeWorker:
    def __init__(self, config: Config):
        self.config = config
        self.model = None  # type: ReversiModel
        self.total_steps = 0
        self.loaded_filenames = set()
        self.loaded_data = {}
        self.dataset = None
        self.optimizer = None
        self.learning_rate = 0.2

    def start(self):
        self.model, self.total_steps = self.load_model()
        self.training()

    def training(self):
        last_generation_step = last_save_step = 0
        min_data_size_to_learn = self.config.trainer.min_data_size_to_learn
        self.load_play_data()

        while True:
            if self.dataset_size < min_data_size_to_learn:
                logger.info(f"dataset_size={self.dataset_size} is less than {min_data_size_to_learn}")
                sleep(60)
                self.load_play_data()
                continue
            self.update_learning_rate()
            steps = self.train_epoch(self.config.trainer.epoch_to_checkpoint)
            self.total_steps = steps

            if last_save_step + self.config.trainer.save_model_steps < self.total_steps:
                self.save_current_model()
                last_save_step = self.total_steps

                # TODO: maybe this reload timing is not consistent with the AZ paper...
                self.load_play_data()

    def generate_train_data(self, batch_size):
        env = ReversiEnv()
        # The AZ paper doesn't leverage the symmetric observation data augmentation. But it is nice to use it if we can.
        symmetric_n = env.rotate_flip_op_count

        while True:
            orig_data_size = self.dataset.size
            data_size = orig_data_size * symmetric_n if symmetric_n > 1 else orig_data_size

            x, lm, y1, y2 = [], [], [], []
            for _ in range(batch_size):
                n = randint(0, data_size - 1)
                orig_n = n // symmetric_n if symmetric_n > 1 else n

                file_name, offset = self.dataset.locate(orig_n)

                state, policy, legal_moves, z = self.loaded_data[file_name][offset]
                state = env.decompress_ob(state)

                if symmetric_n > 1:
                    op = n % symmetric_n
                    state = env.rotate_flip_ob(state, op)
                    policy = env.rotate_flip_pi(policy, op)
                    legal_moves = env.rotate_flip_pi(legal_moves, op)

                state = np.transpose(state, [1, 2, 0])
                x.append(state)
                lm.append(legal_moves)
                y1.append(policy)
                y2.append([z])

            x = np.asarray(x)
            lm = np.asarray(lm)
            y1 = np.asarray(y1)
            y2 = np.asarray(y2)
            yield x, lm, y1, y2

    def train_epoch(self, epochs):
        tc = self.config.trainer
        epoch = 0
        start_time = time.time()
        for x, legal_moves, policy, value in self.generate_train_data(tc.batch_size):
            step = self.model.train(x, legal_moves, policy, value, self.learning_rate)
            if step % tc.save_model_steps == 0:
                global_step, policy_loss, value_loss, reg_loss, total_loss = self.model.train_summary(x, legal_moves, policy, value)
                logger.info(f"step={global_step} loss policy {policy_loss:0.3} value {value_loss:0.3} reg {reg_loss:0.3} total {total_loss:0.3} time {int(time.time() - start_time)}s")
                start_time = time.time()
            epoch += 1
            if epoch * tc.batch_size > self.dataset_size * epochs * 8:
                break

        return global_step

    def update_learning_rate(self):

        for this_lr, till_step in self.config.trainer.lr_schedule:
            if self.total_steps < till_step:
                lr = this_lr
                break
        self.learning_rate = lr
        logger.debug(f"total step={self.total_steps}, set learning rate to {lr}")

    def save_current_model(self):
        self.model.save(self.config.resource.model_dir, self.total_steps//100)

    @property
    def dataset_size(self):
        if self.dataset is None:
            return 0
        return self.dataset.size

    def load_model(self):
        from reversi_zero.agent.model import ReversiModel
        model = ReversiModel(self.config)
        model.build_train(self.config.resource.tensor_log_dir)
        model.create_session()
        logger.debug(f"loading model")
        steps = model.load(self.config.resource.model_dir)
        if steps is None:
            steps = 0
        return model, steps

    def load_play_data(self):
        with MkdirLockFile(self.config.resource.play_data_dir):
            filenames = get_game_data_filenames(self.config.resource)
            updated = False
            for filename in filenames:
                if filename in self.loaded_filenames:
                    continue
                self.load_data_from_file(filename)
                updated = True

            for filename in (self.loaded_filenames - set(filenames)):
                self.unload_data_of_file(filename)
                updated = True

            if updated:
                self.dataset = DataSet(self.loaded_data)
                logger.debug(f"updating training dataset size {self.dataset_size}")

    def load_data_from_file(self, filename):
        try:
            logger.debug(f"loading data from {filename}")
            data = read_game_data_from_file(filename)
            self.loaded_data[filename] = data
            self.loaded_filenames.add(filename)
        except Exception as e:
            logger.warning(str(e))

    def unload_data_of_file(self, filename):
        logger.debug(f"removing data about {filename} from training set")
        self.loaded_filenames.remove(filename)
        if filename in self.loaded_data:
            del self.loaded_data[filename]
