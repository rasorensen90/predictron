'''
A TensorFlow implementation of
The Predictron: End-To-End Learning and Planning
Silver et al.
https://arxiv.org/abs/1612.08810
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import tensorflow as tf
from six.moves import range


logging.basicConfig()
logger = logging.getLogger('predictron')
logger.setLevel(logging.INFO)


class Predictron(object):
  def __init__(self, config):
    # self.inputs = tf.placeholder(tf.float32, shape=[None, config.maze_size, config.maze_size, 1])
    # self.targets = tf.placeholder(tf.float32, shape=[None, 20])

    self.input_shape = [None, config.maze_size, config.maze_size, 1]
    self.target_shape = [None, config.maze_size]

    self.maze_size = config.maze_size
    self.max_depth = config.max_depth
    self.learning_rate = config.learning_rate
    self.max_grad_norm = config.max_grad_norm

    # Tensor rewards with shape [batch_size, max_depth + 1, maze_size]
    self.rewards = None
    # Tensor gammas with shape [batch_size, max_depth + 1, maze_size]
    self.gammas = None
    # Tensor lambdas with shape [batch_size, max_depth, maze_size]
    self.lambdas = None
    # Tensor values with shape [batch_size, max_depth + 1, maze_size]
    self.values = None
    # Tensor  preturns with shape [batch_size, max_depth + 1, maze_size]
    self.preturns = None
    # Tensor lambda_preturns with shape [batch_size, maze_size]
    self.lambda_preturns = None

    self.build()

  def build(self):
    logger.info('Buidling Predictron.')
    self.build_model()
    self.build_loss()

    logger.info('Trainable variables:')
    logger.info('*' * 30)
    for var in tf.trainable_variables():
      logger.info(var.op.name)
    logger.info('*' * 30)

  def iter_func(self, state):
    # sc = predictron_arg_scope()

    value_net = tf.keras.models.Sequential([
          tf.keras.layers.Dense(32, activation='relu')(state),
          tf.keras.layers.BatchNormalization(axis=1),
          tf.keras.layers.Dense(self.maze_size),
        ])
    
    net_ = tf.keras.models.Sequential([
          tf.keras.layers.Conv2D(32, [3,3], activation='relu'),
          tf.keras.layers.BatchNormalization(axis=1),
        ])
    
    net_flatten = tf.keras.layers.Flatten()(net_)
    
    reward_net = tf.keras.models.Sequential([
          tf.keras.layers.Dense(32, activation='relu')(net_flatten),
          tf.keras.layers.BatchNormalization(axis=1),
          tf.keras.layers.Dense(self.maze_size),
        ])
    
    gamma_net = tf.keras.models.Sequential([
          tf.keras.layers.Dense(32, activation='relu')(net_flatten),
          tf.keras.layers.BatchNormalization(axis=1),
          tf.keras.layers.Dense(self.maze_size, activation='sigmoid'),
        ])
    
    lambda_net = tf.keras.models.Sequential([
          tf.keras.layers.Dense(32, activation='relu')(net_flatten),
          tf.keras.layers.BatchNormalization(axis=1),
          tf.keras.layers.Dense(self.maze_size, activation='sigmoid'),
        ])
    
    net = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32,[3,3], activation='relu')(net_),
        tf.keras.layers.BatchNormalization(axis=1),
        tf.keras.layers.Conv2D(32,[3,3], activation='relu'),
        tf.keras.layers.BatchNormalization(axis=1),
        ])

    return net, reward_net, gamma_net, lambda_net, value_net

  def build_model(self):
    state = tf.keras.models.Sequential([
          tf.keras.layers.Conv2D(32, [3,3], activation='relu'),
          tf.keras.layers.BatchNormalization(axis=1),
          tf.keras.layers.Conv2D(32, [3,3], activation='relu'),
          tf.keras.layers.BatchNormalization(axis=1)
        ])
    

    iter_template = tf.make_template('iter', self.iter_func, unique_name_='iter')

    rewards_arr = []
    gammas_arr = []
    lambdas_arr = []
    values_arr = []

    for k in range(self.max_depth):
      state, reward, gamma, lambda_, value = iter_template(state)
      rewards_arr.append(reward)
      gammas_arr.append(gamma)
      lambdas_arr.append(lambda_)
      values_arr.append(value)

    _, _, _, _, value = iter_template(state)
    # K + 1 elements
    values_arr.append(value)

    # [batch_size, K * maze_size]
    self.rewards = tf.pack(rewards_arr, axis=1)
    # [batch_size, K, maze_size]
    self.rewards = tf.reshape(self.rewards, [-1, self.max_depth, self.maze_size])
    # [batch_size, K + 1, maze_size]
    self.rewards = tf.concat_v2(values=[tf.zeros(shape=[-1, 1, self.maze_size], dtype=tf.float32), self.rewards],
                                axis=1, name='rewards')

    # [batch_size, K * maze_size]
    self.gammas = tf.pack(gammas_arr, axis=1)
    # [batch_size, K, maze_size]
    self.gammas = tf.reshape(self.gammas, [None, self.max_depth, self.maze_size])
    # [batch_size, K + 1, maze_size]
    self.gammas = tf.concat_v2(values=[tf.ones(shape=[-1, 1, self.maze_size], dtype=tf.float32), self.gammas],
                               axis=1, name='gammas')

    # [batch_size, K * maze_size]
    self.lambdas = tf.pack(lambdas_arr, axis=1)
    # [batch_size, K, maze_size]
    self.lambdas = tf.reshape(self.lambdas, [-1, self.max_depth, self.maze_size])

    # [batch_size, (K + 1) * maze_size]
    self.values = tf.pack(values_arr, axis=1)
    # [batch_size, K + 1, maze_size]
    self.values = tf.reshape(self.values, [-1, (self.max_depth + 1), self.maze_size])

    self.build_preturns()
    self.build_lambda_preturns()

  def build_preturns(self):
    ''' Eqn (2) '''

    g_preturns = []
    # for k = 0, g_0 = v[0], still fits.
    for k in range(self.max_depth, -1, -1):
      g_k = self.values[:, k, :]
      for kk in range(k, 0, -1):
        g_k = self.rewards[:, kk, :] + self.gammas[:, kk, :] * g_k
      g_preturns.append(g_k)
    # reverse to make 0...K from K...0
    g_preturns = g_preturns[::-1]
    self.g_preturns = tf.pack(g_preturns, axis=1, name='preturns')
    self.g_preturns = tf.reshape(self.g_preturns, [-1, self.max_depth + 1, self.maze_size])

  def build_lambda_preturns(self):
    ''' Eqn (4) '''
    g_k = self.values[:, -1, :]
    for k in range(self.max_depth - 1, -1, -1):
      g_k = (1 - self.lambdas[:, k, :]) * self.values[:, k, :] + \
            self.lambdas[:, k, :] * (self.rewards[:, k + 1, :] + self.gammas[:, k + 1, :] * g_k)
    self.g_lambda_preturns = g_k

  def build_loss(self):
    with tf.variable_scope('loss'):
      # Loss Eqn (5)
      # [batch_size, 1, maze_size]
      self.targets_tiled = tf.expand_dims(self.targets, 1)
      # [batch_size, K + 1, maze_size]
      self.targets_tiled = tf.tile(self.targets_tiled, [1, self.max_depth + 1, 1])
      self.loss_preturns = losses.mean_squared_error(self.g_preturns, self.targets_tiled, scope='preturns')
      losses.add_loss(self.loss_preturns)
      tf.summary.scalar('loss_preturns', self.loss_preturns)
      # Loss Eqn (7)
      self.loss_lambda_preturns = losses.mean_squared_error(
        self.g_lambda_preturns, self.targets, scope='lambda_preturns')
      losses.add_loss(self.loss_lambda_preturns)
      tf.summary.scalar('loss_lambda_preturns', self.loss_lambda_preturns)
      self.total_loss = losses.get_total_loss(name='total_loss')
