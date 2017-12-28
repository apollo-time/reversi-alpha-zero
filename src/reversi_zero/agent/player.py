# -*- coding: utf-8 -*-

import numpy as np
import random
import time
from asyncio.queues import Queue, QueueEmpty
from collections import namedtuple
from logging import getLogger
import asyncio
import copy

from reversi_zero.agent.api import ReversiModelAPI

logger = getLogger(__name__)

QueueItem = namedtuple("QueueItem", "node state legal_moves future")


class SelfPlayer(object):
    def __init__(self, make_sim_env_fn, config, model, play_config=None):
        self.game_tree = GameTree(make_sim_env_fn=make_sim_env_fn, config=config, model=model, play_config=play_config)

    def prepare(self, root_env, dir_noise):
        self.game_tree.expand_root(root_env=root_env, dir_noise=dir_noise)

    def think(self, tau=0):
        return self.game_tree.mcts_and_play(tau)

    def play(self, act):
        self.game_tree.keep_only_subtree(act)


class EvaluatePlayer(SelfPlayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def play(self, act, env):
        self.game_tree.keep_only_subtree(act)
        if not self.game_tree.root_node.expanded:
            # possible that opposite plays an action which I haven't searched yet.
            self.game_tree.expand_root(root_env=env, dir_noise=False)


class GameTree(object):
    def __init__(self, make_sim_env_fn, config=None, play_config=None, model=None):
        self.make_sim_env_fn = make_sim_env_fn
        self.config = config
        self.play_config = play_config or self.config.play
        self.root_node = Node(self.play_config.c_puct)
        self.model = model
        self.prediction_queue = Queue(self.play_config.prediction_queue_size)
        self.api = ReversiModelAPI(self.config, self.model)

        self.loop = asyncio.get_event_loop()
        self.running_simulation_num = 0
        self.expanding_nodes = set()
        self.locker = asyncio.Lock()

    def expand_root(self, root_env, dir_noise=None):
        ps, vs = self.api.predict(np.asarray(root_env.observation), np.asarray(root_env.legal_moves))
        self.root_node.expand_and_evaluate(ps, vs, root_env.legal_moves)
        if dir_noise:
            self.root_node.add_dirichlet_noise(self.play_config.noise_eps, self.play_config.dirichlet_alpha)

    def mcts_and_play(self, tau):
        self.mcts()
        return self.play(tau)

    def keep_only_subtree(self, action):
        root_node = self.root_node.child_by_value(action)
        if root_node == None:
            root_node = self.root_node
        assert root_node is not None, f'root node has {len(self.root_node.children)} child  action = {action}'
        self.root_node = root_node

    def mcts(self):
        self.running_simulation_num = self.play_config.simulation_num_per_move
        coroutine_list = []
        start_time = time.time()
        for it in range(self.play_config.simulation_num_per_move):
            coroutine_list.append(self.simulate())
        coroutine_list.append(self.prediction_worker())
        self.loop.run_until_complete(asyncio.gather(*coroutine_list))
        # print('search mcts in time %.2f' % (time.time() - start_time))
    async def simulate(self):
        leaf_v = await self.simulate_internal()
        await self.prediction_queue.put(None)
        return leaf_v

    async def simulate_internal(self):
        assert self.root_node.expanded

        virtual_loss = self.config.play.virtual_loss
        env = self.make_sim_env_fn()
        cur_node = self.root_node
        while True:
            next_node = await self.select_next_or_expand(env, cur_node)
            if next_node is None:
                # cur_node is expanded leaf node
                leaf_node = cur_node
                v = leaf_node.v
                break
            env.step(next_node.value)
            if env.done:
                leaf_node = next_node
                v = 1 if env.last_player_wins else -1 if env.last_player_loses else 0
                v = -v
                leaf_node.v = float(v)
                break
            # select next node
            cur_node = next_node
        # backup
        cur_node = leaf_node
        while cur_node is not self.root_node:
            v = -v  # important: reverse v
            parent = cur_node.parent
            with await parent.locker:
                parent.backup(v, virtual_loss, cur_node.sibling_index)

            cur_node = parent
        return -v  # v for root node

    async def select_next_or_expand(self, env, node):
        with await node.locker:
            if node.expanded:
                # select node
                if node.passed:
                    return node.children[0]
                ci = random.choice(node.best_children_indices)
                next_node = node.children[ci]
                virtual_loss = self.config.play.virtual_loss
                node.add_virtual_loss(virtual_loss, next_node.sibling_index)
                return next_node
            # expand node
            ob, legal_moves, rotate_flip_op = np.asarray(env.observation), np.asarray(env.legal_moves), None
            env_legal_moves = legal_moves
            if env.rotate_flip_op_count > 0:
                rotate_flip_op = random.randint(0, env.rotate_flip_op_count - 1)
                ob = env.rotate_flip_ob(ob, rotate_flip_op)
                legal_moves = env.rotate_flip_pi(legal_moves, rotate_flip_op)

            future = await self.predict(ob, legal_moves)
            await future
            p, v = future.result()

            if rotate_flip_op is not None:
                p = env.counter_rotate_flip_pi(p, rotate_flip_op)

            node.expand_and_evaluate(p, v, env_legal_moves)
            return None

    async def prediction_worker(self):
        q = self.prediction_queue
        while self.running_simulation_num > 0:
            item_list = []
            item = await q.get()
            if item is None:
                self.running_simulation_num -= 1
            else:
                item_list.append(item)
            while not q.empty():
                try:
                    item = q.get_nowait()
                    if item is None:
                        self.running_simulation_num -= 1
                        continue
                    item_list.append(item)
                except QueueEmpty:
                    break
            if len(item_list) == 0:
                continue
            start_time = time.time()
            data = np.array([x.state for x in item_list])
            legal_moves = np.array([x.legal_moves for x in item_list])
            policy_ary, value_ary = self.api.predict(data, legal_moves)  # policy_ary: [n, 64], value_ary: [n, 1]
            for p, v, item in zip(policy_ary, value_ary, item_list):
                item.future.set_result((p, v))
            # print('prediction worker process %d stats in time %.2f' % (len(item_list), time.time() - start_time))

    async def predict(self, x, legal_moves):
        future = self.loop.create_future()
        item = QueueItem(self, x, legal_moves, future)
        await self.prediction_queue.put(item)
        return future

    # those illegal actions are with full_N == 0, so won't be played
    def play(self, tau):
        if self.root_node.passed:
            pi = np.zeros([self.root_node._full_n_size])
            act = 64
        else:
            N = self.root_node.full_N
            if abs(tau-1) < 1e-10:
                pi = N / np.sum(N)
                act = np.random.choice(range(len(pi)), p=pi)
                assert pi[act] > 0
            else:
                assert abs(tau) < 1e-10, f'tau={tau}(expected to be either 0 or 1 only)'
                act = random.choice(np.argwhere(abs(N - np.amax(N)) < 1e-10).flatten().tolist())
                pi = np.zeros([len(N)])
                pi[act] = 1

        # the paper says, AGZ resigns if both root value and best child value are lower than threshold
        # TODO: is it v or Q or Q+U to check?
        root_v = self.root_node.v
        # child'v is opponent's winning rate, need to reverse
        # Note that root_node.children are only for those legal action.
        children_v = [-child.v for child in self.root_node.children]
        if len(children_v) > 0:
            best_child_v = np.max(children_v)
        else:
            best_child_v = root_v  # trick. Since it is for resign_check only, it works to let be root_v.
        values_of_resign_check = (root_v, best_child_v)

        return int(act), pi, values_of_resign_check


class Node(object):
    def __init__(self, c_puct, parent=None, sibling_index=None, value=None):
        self.children = None
        self._parent = parent

        self._c_puct = c_puct
        self._sibling_index = sibling_index
        self._value = value  # corresponding "action" of env

        self.p = None
        self.W = None
        self.Q = None
        self.N = None
        self.v = 0.

        # below variables are only for speeding up MCTS
        self._sum_n = None
        self._best_children_indices = None
        self._full_n_size = None
        self.locker = asyncio.Lock()

    # given the real meaning of node.value, full_N is actually N for every "action" of env
    @property
    def full_N(self):
        assert self.expanded

        assert np.sum(self.N) > 0, f'full_N is called with self.N={self.N}'

        ret = np.zeros([self._full_n_size])
        for node in self.children:
            ret[node.value] = self.N[node.sibling_index]

        assert abs(np.sum(self.N) - np.sum(ret)) < 1e-10
        return ret

    @property
    def expanded(self):
        return self.children is not None

    @property
    def passed(self):
        return len(self.p) == 0

    @property
    def value(self):
        return self._value

    @property
    def sibling_index(self):
        return self._sibling_index

    @property
    def parent(self):
        return self._parent

    def child_by_value(self, value):
        return next((child for child in self.children if child.value == value), None)

    def expand_and_evaluate(self, p, v, legal_moves):

        self.p = p[legal_moves == 1]  # this.p is (typically much) shorter than p
        #assert 1 - np.sum(self.p) < 1e-2, f'invalid legal moves {legal_moves} or pi {self.p}'
        assert 0 <= len(self.p) < len(legal_moves)
        self.v = v
        self.W = np.zeros([len(self.p)])
        self.Q = np.zeros([len(self.p)])
        self.N = np.zeros([len(self.p)])

        actions = (i for i,v in enumerate(legal_moves) if v == 1)
        self.children = [Node(c_puct=self._c_puct, parent=self, sibling_index=i, value=a)
                         for i,a in enumerate(actions)]
        if len(self.children) == 0:
            self.children = [Node(c_puct=self._c_puct, parent=self, sibling_index=0, value=64)]

        self._sum_n = 0
        self._best_children_indices = None
        self._full_n_size = len(legal_moves)

    def add_dirichlet_noise(self, eps, alpha):
        self.p = (1-eps)*self.p + eps*np.random.dirichlet([alpha]*len(self.p))
        self._best_children_indices = None

    def add_virtual_loss(self, virtual_loss, child):
        self.N[child] += virtual_loss
        self.W[child] -= virtual_loss
        self.Q[child] = self.W[child] / self.N[child]
        assert self.N[child] > 0, f'N[{child}]={self.N[child]}'

        self._sum_n += virtual_loss
        self._best_children_indices = None

    def substract_virtual_loss(self, virtual_loss, child):
        self.N[child] -= virtual_loss
        self.W[child] += virtual_loss
        self.Q[child] = self.W[child] / self.N[child]
        assert self.N[child] >= 0, f'N[{child}]={self.N[child]}'

        self._sum_n -= virtual_loss
        self._best_children_indices = None

    def backup(self, v, virtual_loss, child):
        if self.passed:
            return
        self.N[child] += 1 - virtual_loss
        self.W[child] += v + virtual_loss
        self.Q[child] = self.W[child] / self.N[child]
        assert self.N[child] > 0, f'N[{child}]={self.N[child]}'

        self._sum_n += 1 - virtual_loss
        self._best_children_indices = None

    @property
    def best_children_indices(self):
        if self._best_children_indices is None:
            if len(self.p) == 1:
                self._best_children_indices = [0]
            else:
                sqrt_sum_n = np.sqrt(self._sum_n)
                v = self.Q + self._c_puct * self.p * sqrt_sum_n / (1 + self.N)
                self._best_children_indices = np.argwhere(abs(v-np.amax(v)) < 1e-10).flatten().tolist()
                if len(self._best_children_indices) == 0:
                    self._best_children_indices = [0]
        return self._best_children_indices
