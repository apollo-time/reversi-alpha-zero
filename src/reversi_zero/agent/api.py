from reversi_zero.config import Config
import numpy as np

class ReversiModelAPI:
    def __init__(self, config: Config, agent_model):
        """

        :param config:
        :param reversi_zero.agent.model.ReversiModel agent_model:
        """
        self.config = config
        self.agent_model = agent_model

    def predict(self, x, legal_moves):
        assert x.ndim in (3, 4), f'{x.ndim}'
        orig_x = x
        if False:
            s = 1 if x.ndim == 3 else x.shape[0]
            policy, value = np.ones([s, 64], np.float32) / 65, np.zeros([s,1], np.float32)
        else:
            if x.ndim == 3:
                x = x.reshape(1, -1, 8, 8)
            if legal_moves.ndim == 1:
                legal_moves = legal_moves.reshape(1, -1)
            x = np.transpose(x, [0,2,3,1])
            policy, value = self.agent_model.predict(x, legal_moves)

        if orig_x.ndim == 3:
            return policy[0], value[0,0]
        else:
            return policy, value[:,0]


