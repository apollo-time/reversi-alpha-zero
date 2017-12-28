import hashlib
import json
import os
from logging import getLogger
# noinspection PyPep8Naming
import tensorflow as tf
from lockfile.mkdirlockfile import MkdirLockFile
slim = tf.contrib.slim

from reversi_zero.config import Config

logger = getLogger(__name__)


class ReversiModel:
    def __init__(self, config: Config):
        self.config = config
        self.latest_checkpoint = None
        self.model_name = 'reversi_v1'
        self.build()

    def build(self):
        mc = self.config.model
        self.x_placehoder = x = tf.placeholder(tf.float32, shape=(None,8, 8, mc.history_len*2), name='board')  # [own(8x8), enemy(8x8)] * history_len
        self.legal_moves_placehoder = tf.placeholder(tf.float32, shape=(None, 64), name='legal_moves')
        self.phase_train_placeholder = phase_train = tf.placeholder(tf.bool, name='phase_train')
        batch_norm_params = {
            # Decay for the moving averages.
            'decay': 0.995,
            # epsilon to prevent 0s in variance.
            'epsilon': 0.001,
            # calculate moving average or using exist one
            'is_training': phase_train,
            'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
            'updates_collections': None,
        }
        with tf.variable_scope("reversi"):
            # Set weight_decay for weights in Conv and FC layers.
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_initializer=slim.xavier_initializer_conv2d(uniform=True),
                                weights_regularizer=slim.l2_regularizer(mc.l2_reg),
                                normalizer_fn=slim.batch_norm,
                                activation_fn=tf.nn.relu,
                                normalizer_params=batch_norm_params):
                with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                    x = slim.conv2d(x, mc.cnn_filter_num, mc.cnn_filter_size, stride=1, scope='conv1')
                    for i in range(mc.res_layer_num):
                        x = self._build_residual_block(x, scope='residual%d' % (i+1))
                    res_out = x
                    # for policy output
                    x = slim.conv2d(res_out, 2, 1, stride=1, scope='policy_conv1')
                    x = slim.flatten(x)
                    x = slim.fully_connected(x, mc.action_count, activation_fn=None, scope="policy_fn1")
                    self.policy_logits = x - (1 - self.legal_moves_placehoder) * 1000
                    self.policy_out = tf.nn.softmax(self.policy_logits, -1, name='policy')
                    # for value output
                    x = slim.conv2d(res_out, 1, 1, stride=1, scope='value_conv1')
                    x = slim.flatten(x)
                    x = slim.fully_connected(x, mc.value_fc_size, scope="value_fn1")
                    self.value_out = slim.fully_connected(x, 1, activation_fn=tf.nn.tanh, scope="value_fn2")

    def build_train(self, log_dir):
        mc = self.config.model
        self.policy_placehoder = x = tf.placeholder(tf.float32, shape=(None, mc.action_count),
                                                    name='policy_placeholder')
        self.value_placehoder = x = tf.placeholder(tf.float32, shape=(None, 1),
                                                    name='value_placeholder')
        self.learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.regularization_losses = sum(regularization_losses)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.policy_placehoder, logits=self.policy_logits, name='cross_entropy_per_example')
        legal_moves_valid = tf.reduce_max(self.legal_moves_placehoder, axis=-1)
        cross_entropy = tf.multiply(cross_entropy, legal_moves_valid)
        self.policy_loss = tf.reduce_mean(cross_entropy, name='policy_loss')
        self.value_losses = tf.losses.mean_squared_error(labels=self.value_placehoder, predictions=self.value_out, scope='value_losses')
        self.total_losses = self.policy_loss + self.value_losses + self.regularization_losses
        self.global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder)
        self.train_op = optimizer.minimize(self.total_losses, global_step=self.global_step)

        tf.summary.scalar('loss/regularization', self.regularization_losses)
        tf.summary.scalar('loss/policy', self.policy_loss)
        tf.summary.scalar('loss/value', self.value_losses)
        tf.summary.scalar('loss/total', self.total_losses)
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(log_dir)

    def create_session(self):
        self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                per_process_gpu_memory_fraction=self.config.model.gpu_mem_frac,
                allow_growth=None,
            )
        )
        self.sess = sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

    def _build_residual_block(self, x, scope):
        mc = self.config.model
        in_x = x
        with tf.variable_scope(scope):
            x = slim.conv2d(x, mc.cnn_filter_num, mc.cnn_filter_size, stride=1, scope='conv1')
            x = slim.conv2d(x, mc.cnn_filter_num, mc.cnn_filter_size, activation_fn=None, stride=1, scope='conv2')
            x = in_x + x
            x = tf.nn.relu(x)

        return x

    def predict(self, x, legal_moves):
        return self.sess.run([self.policy_out, self.value_out], feed_dict={self.x_placehoder:x, self.legal_moves_placehoder:legal_moves, self.phase_train_placeholder:False})

    def train(self, x, legal_moves, policy, value, learning_rate):
        global_step, _ = self.sess.run([self.global_step, self.train_op],
                                       feed_dict={self.x_placehoder:x, self.legal_moves_placehoder:legal_moves, self.policy_placehoder:policy, self.value_placehoder:value,
                                                  self.phase_train_placeholder:True, self.learning_rate_placeholder:learning_rate})
        return global_step

    def train_summary(self, x, legal_moves, policy, value):
        global_step, policy_loss, value_loss, reg_loss, total_loss, summary_str = self.sess.run([self.global_step, self.policy_loss, self.value_losses, self.regularization_losses, self.total_losses, self.summary_op],
                                       feed_dict={self.x_placehoder: x, self.legal_moves_placehoder: legal_moves,
                                                  self.policy_placehoder: policy, self.value_placehoder: value, self.phase_train_placeholder:False})
        self.summary_writer.add_summary(summary_str, global_step=global_step)
        return global_step, policy_loss, value_loss, reg_loss, total_loss

    @staticmethod
    def fetch_digest(weight_path):
        if os.path.exists(weight_path):
            m = hashlib.sha256()
            with open(weight_path, "rb") as f:
                m.update(f.read())
            return m.hexdigest()

    def load(self, save_path):
        with MkdirLockFile(save_path):
            latest_checkpoint = tf.train.latest_checkpoint(save_path)
            if latest_checkpoint is None:
                return None
            if self.latest_checkpoint == latest_checkpoint:
                return None
            logger.debug(f"loading model from {latest_checkpoint}")
            self.saver.restore(self.sess, latest_checkpoint)
            self.latest_checkpoint = latest_checkpoint

    def save(self, save_path, step):
        logger.debug(f"save model to {save_path}")
        with MkdirLockFile(save_path):
            checkpoint_path = os.path.join(save_path, 'model-%s.ckpt' % self.model_name)
            self.saver.save(self.sess, checkpoint_path, global_step=step, write_meta_graph=False)
            self.latest_checkpoint = checkpoint_path
            logger.debug(f"saved model path {checkpoint_path}")
            return checkpoint_path
