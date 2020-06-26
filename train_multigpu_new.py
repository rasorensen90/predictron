'''
Training the predictron in multiple GPUs.

Modified from Tensorflow/models/tutorials/image/cifar10/cifar10_multi_gpu_train.py

MultiGPUs sync gradient descents on a single machine.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import logging
import os
import threading
import time
import absl

import numpy as np
import six.moves.queue as queue
import tensorflow as tf
from six.moves import range

from maze import MazeGenerator
from predictron import Predictron

FLAGS = absl.flags.FLAGS

absl.flags.DEFINE_string('train_dir', './ckpts/predictron_train',
                       'dir to save checkpoints and TB logs')
absl.flags.DEFINE_integer('max_steps', 10000000, 'num of batches')
absl.flags.DEFINE_integer('num_gpus', 8, 'num of GPUs to use')
absl.flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')

absl.flags.DEFINE_integer('batch_size', 128, 'batch size')
absl.flags.DEFINE_integer('maze_size', 20, 'size of maze (square)')
absl.flags.DEFINE_float('maze_density', 0.3, 'Maze density')
absl.flags.DEFINE_integer('max_depth', 16, 'maximum model depth')
absl.flags.DEFINE_float('max_grad_norm', 10., 'clip grad norm into this value')
absl.flags.DEFINE_boolean('log_device_placement', False,
                        """Whether to log device placement.""")
absl.flags.DEFINE_integer('num_threads', 10, 'num of threads used to generate mazes.')


logging.basicConfig()
logger = logging.getLogger('multigpu_train')
logger.setLevel(logging.INFO)


def tower_loss(scope, maze_ims, maze_labels, config):
  '''
  Computer the loss for each GPU tower.
  Args:
    scope: tower scope
    maze_ims: Tensor of [batch_size, maze_size, maze_size, 1] of maze images
    maze_labels: Tensor of [batch_size, maze_size] for target label of the connection of diagonal elements
    config: configuration of the predictron hyperparameters
  Returns:
    total_loss to optimize, preturns regression loss and \lambda-preturn loss
  '''
  model = Predictron(maze_ims, maze_labels, config)
  loss_preturns = model.loss_preturns
  loss_lambda_preturns = model.loss_lambda_preturns
  losses = tf.get_collection('losses', scope)
  total_loss = tf.add_n(losses, name='total_loss')
  return total_loss, loss_preturns, loss_lambda_preturns


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat_v2(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train():
  '''Training function'''

  # The large batch is divided arcoss all towers
  if FLAGS.batch_size % FLAGS.num_gpus != 0:
    raise ValueError('batch_size should be divisible by num_gpus, bs = {}, num_gpus = {}'.format(
      FLAGS.batch_size, FLAGS.num_gpus))

  # Data queue
  maze_queue = queue.Queue(100)

  def maze_generator():
    # maze generator thread function
    maze_gen = MazeGenerator(
      height=FLAGS.maze_size,
      width=FLAGS.maze_size,
      density=FLAGS.maze_density)

    while True:
      maze_ims, maze_labels = maze_gen.generate_labelled_mazes(FLAGS.batch_size)
      maze_queue.put((maze_ims, maze_labels))

  # Start a bunch of threads to generate maze data
  for thread_i in range(FLAGS.num_threads):
    t = threading.Thread(target=maze_generator)
    t.start()

  config = FLAGS
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
      'global_step', [],
      initializer=tf.constant_initializer(0), trainable=False)

    # optimizer
    opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
    # placeholders for the large batch
    maze_ims_ph = tf.placeholder(tf.int32, shape=[None, FLAGS.maze_size, FLAGS.maze_size, 1])
    maze_labels_ph = tf.placeholder(tf.int32, shape=[None, FLAGS.maze_size])
    # split the large batch arcoss all towers
    maze_ims_splits = tf.split(0, FLAGS.num_gpus, maze_ims_ph)
    maze_labels_splits = tf.split(0, FLAGS.num_gpus, maze_labels_ph)
    # Calculate the gradients for each model tower.
    tower_grads = []
    for i in range(FLAGS.num_gpus):
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('%s_%d' % ('predictron', i)) as scope:
          # Calculate the loss for one tower of the predictron model. This function
          # constructs the entire predictron model but shares the variables across
          # all towers.
          loss, loss_preturns, loss_lambda_preturns = tower_loss(
            scope,
            maze_ims_splits[i],
            maze_labels_splits[i],
            config)

          # Reuse variables for the next tower.
          tf.get_variable_scope().reuse_variables()

          # Retain the summaries from the final tower.
          summary_merged = tf.summary.merge_all()

          # Calculate the gradients for the batch of data on this predictron tower.
          grads = opt.compute_gradients(loss)

          # Keep track of the gradients across all towers.
          tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grad_vars = average_gradients(tower_grads)
    grads, vars = zip(*grad_vars)
    grads_clipped, _ = tf.clip_by_global_norm(grads, FLAGS.max_grad_norm)
    grad_vars = zip(grads_clipped, vars)
    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grad_vars, global_step=global_step)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='predictron_0')
    update_op = tf.group(*update_ops)
    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, update_op)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())
    # summary op
    summary_op = tf.identity(summary_merged)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)
    # directory to save model checkpoints and events
    train_dir = os.path.join(FLAGS.train_dir, 'max_steps_{}'.format(FLAGS.max_depth))
    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

    for step in range(FLAGS.max_steps):
      start_time = time.time()
      # get data from the data queue
      maze_ims_np, maze_labels_np = maze_queue.get()
      # session run
      _, loss_value, loss_preturns_val, loss_lambda_preturns_val, summary_str = sess.run(
        [train_op, loss, loss_preturns, loss_lambda_preturns, summary_op],
        feed_dict={
          maze_ims_ph: maze_ims_np,
          maze_labels_ph: maze_labels_np
        })
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / FLAGS.num_gpus

        format_str = (
          '%s: step %d, loss = %.4f, loss_preturns = %.4f, loss_lambda_preturns = %.4f (%.1f examples/sec; %.3f '
          'sec/batch)')
        logger.info(
          format_str % (datetime.datetime.now(), step, loss_value, loss_preturns_val, loss_lambda_preturns_val,
                        examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):
  train()


if __name__ == '__main__':
  tf.app.run()
