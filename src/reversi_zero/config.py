import os


def _project_dir():
    d = os.path.dirname
    return d(d(d(os.path.abspath(__file__))))


def _data_dir():
    return os.path.join(_project_dir(), "data")


class Config:
    def __init__(self):
        self.opts = Options()
        self.resource = ResourceConfig()
        self.gui = GuiConfig()

        self.model = ModelConfig()
        self.play = PlayConfig()
        self.play_data = PlayDataConfig()
        self.trainer = TrainerConfig()


class Options:
    new = False


class PlayDataConfig:
    def __init__(self):
        # AGZ paper says: "... from the most recent 500,000 games of self-play.
        # But I reduce to nb_game_in_file * max_file_num == 50,000 due to performance trade-off.
        # Small nb_game_in_file make training data update more frequently, but
        # large max_file_num make larger overhead reading file
        self.nb_game_in_file = 100
        self.max_file_num = 5000


# Some performance notes:
# Using 3 P40 GPUs, 9 self-play processes in total, I am now getting a self-play speed about
# 20 second per game per process, that is about 2 second per game in average.
# In AZ paper, for Go, it's 34 hours / 21m games, that is 0.005 second per game in average.
# So we see a 400x performance.
# Also note I am using 1/8 simulation_num_per_move of AZ.
class PlayConfig:
    def __init__(self):
        # this would be the biggest difference with AZ. I set this due to speed trade-off
        self.simulation_num_per_move = 500  # AZ:800, AGZ:1600
        self.c_puct = 1                     # AZ: UNKNOWN
        self.noise_eps = 0.25               # AZ: same
        self.dirichlet_alpha = 0.03          # AZ: depends on game
        self.change_tau_turn = 10           # AZ: same
        self.virtual_loss = 3               # AZ: UNKNOWN
        self.prediction_queue_size = 120      # AZ: 8
        self.parallel_search_num = 8        # AZ: N/A
        self.prediction_worker_sleep_sec  = 0.0001
        self.wait_for_expanding_sleep_sec = 0.00001
        self.v_resign_check_min_n = 100
        self.v_resign_init = -0.9           # AZ: UNKNOWN
        self.v_resign_delta = 0.01          # AZ: UNKNOWN
        self.v_resign_disable_prop = 0.1    # AZ: same
        self.v_resign_false_positive_fraction_t_max = 0.05  # AZ: same
        # If we don't have a min fraction, then we may have a lower frac, in worst case we will have NO
        # resignation. Than means we will have to train many 1-side games. Not what we want.
        self.v_resign_false_positive_fraction_t_min = 0.04  # AZ: UNKNOWN


# Some performance notes:
# Using a P40 GPU, I am now getting a training speed about 10 minutes / 140 steps,
# that is, 4.3 second per step.
# In AZ paper, for Go, it's 34 hours / 700k steps, that is 0.17 second per step.
# So we see a 25x performance.
# Also note I am using half of batch_size of AZ.
class TrainerConfig:
    def __init__(self):
        self.batch_size = 1024             # AZ: 4096 - I don't have so much GPU memory though
        self.epoch_to_checkpoint = 1
        self.start_total_steps = 0
        # AZ paper says "maintains a single NN that is update continually". That means save_model_steps = 1.
        # However in practice, we want to balance something...
        self.save_model_steps = 100         # AZ: 1?
        self.min_data_size_to_learn = 10    # AZ: N/A
        self.lr_schedule = (  # (learning rate, before step count)
            (0.2,    10000),
            (0.02,   100000),
            (0.002,  200000),
            (0.0002, 9999999999)
        )


# Some notes:
# Combining the performance notes above about self-play and train,
# my ratio of self-play/opt is 25/400=1/16 of AZ.
# This will make it much more easier to stuck in local optimal.
# Will I still train it out to some good level? Let's see...

class ModelConfig:
    history_len = 2
    action_count = 8*8
    cnn_filter_num = 256
    cnn_filter_size = 3
    res_layer_num = 10
    l2_reg = 1e-4
    value_fc_size = 256
    gpu_mem_frac = 0.2


class ResourceConfig:
    def __init__(self):
        self.project_dir = os.environ.get("PROJECT_DIR", _project_dir())
        self.data_dir = os.environ.get("DATA_DIR", _data_dir())
        self.model_dir = os.environ.get("MODEL_DIR", os.path.join(self.data_dir, "model"))
        self.tensor_log_dir = os.environ.get("MODEL_DIR", os.path.join(self.data_dir, "logs"))
        self.model_config_filename = "model_config.json"
        self.model_weight_filename = "model_weight.h5"
        self.model_config_path = os.path.join(self.model_dir, self.model_config_filename)
        self.model_weight_path = os.path.join(self.model_dir, self.model_weight_filename)

        self.use_remote_model = os.environ.get("USE_REMOTE_MODEL")
        self.remote_model_config_path = os.environ.get("MODEL_CONFIG_URL")
        self.remote_model_weight_path = os.environ.get("MODEL_WEIGHT_URL")
        if self.use_remote_model and not self.remote_model_config_path:
            raise Exception("USE_REMOTE_MODEL is True but MODEL_CONFIG_URL is not set!")
        if self.use_remote_model and not self.remote_model_weight_path:
            raise Exception("USE_REMOTE_MODEL is True but MODEL_WEIGHT_URL is not set!")

        self.play_data_dir = os.path.join(self.data_dir, "play_data")
        self.play_data_filename_tmpl = "play_%s.json"

        self.log_dir = os.path.join(self.project_dir, "logs")
        self.main_log_path = os.path.join(self.log_dir, "main.log")

    def create_directories(self):
        dirs = [self.project_dir, self.data_dir, self.model_dir, self.tensor_log_dir, self.play_data_dir, self.log_dir]
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)


class GuiConfig:
    def __init__(self):
        self.window_size = (400, 440)
        self.window_title = "reversi-alpha-zero"


class PlayWithHumanConfig:
    def __init__(self):
        self.simulation_num_per_move = 500
        self.c_puct = 1
        self.parallel_search_num = 8
        self.noise_eps = 0
        self.change_tau_turn = 0

    def update_play_config(self, pc):
        """

        :param PlayConfig pc:
        :return:
        """
        pc.simulation_num_per_move = self.simulation_num_per_move
        pc.c_puct = self.c_puct
        pc.noise_eps = self.noise_eps
        pc.change_tau_turn = self.change_tau_turn
        pc.parallel_search_num = self.parallel_search_num
