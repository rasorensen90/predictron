'''
Training part
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six.moves.queue as queue
from six.moves import range
import datetime
import logging
import os
import threading
import time

import numpy as np
import tensorflow as tf

from absl import app
from absl import flags
from maze import MazeGenerator
from predictron import Predictron

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', './ckpts/predictron_train',
                       'dir to save checkpoints and TB logs')
flags.DEFINE_integer('max_steps', 10000000, 'num of batches')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')

flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('maze_size', 20, 'size of maze (square)')
flags.DEFINE_float('maze_density', 0.3, 'Maze density')
flags.DEFINE_integer('max_depth', 16, 'maximum model depth')
flags.DEFINE_float('max_grad_norm', 10., 'clip grad norm into this value')
flags.DEFINE_boolean('log_device_placement', False,
                        """Whether to log device placement.""")
flags.DEFINE_integer('num_threads', 10, 'num of threads used to generate mazes.')


logging.basicConfig()
logger = logging.getLogger('training')
logger.setLevel(logging.INFO)

def train():
    config = FLAGS
    model = Predictron(config)
    
    
    maze_queue = queue.Queue(100)
    
    def maze_generator():
        maze_gen = MazeGenerator(
            height=FLAGS.maze_size,
            width=FLAGS.maze_size,
            density=FLAGS.maze_density)
        
        while True:
          maze_ims, maze_labels = maze_gen.generate_labelled_mazes(FLAGS.batch_size)
          maze_queue.put((maze_ims, maze_labels))
          break
    for thread_i in range(FLAGS.num_threads):
        t = threading.Thread(target=maze_generator)
        t.start()
    
    for step in range(FLAGS.max_steps):
        start_time = time.time()
        maze_ims_np, maze_labels_np = maze_queue.get()
        

    

    # model.build()

  # loss = model.total_loss
  # loss_preturns = model.loss_preturns
  # loss_lambda_preturns = model.loss_lambda_preturns

  # opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
  # grad_vars = opt.compute_gradients(loss, tf.trainable_variables())
  # grads, vars = zip(*grad_vars)
  # grads_clipped, _ = tf.clip_by_global_norm(grads, FLAGS.max_grad_norm)
  # grad_vars = zip(grads_clipped, vars)
  # apply_gradient_op = opt.apply_gradients(grad_vars, global_step=global_step)

  # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  # update_op = tf.group(*update_ops)
  # # Group all updates to into a single train op.
  # train_op = tf.group(apply_gradient_op, update_op)


  # saver = tf.train.Saver(tf.global_variables())

  # train_dir = os.path.join(FLAGS.train_dir, 'max_steps_{}'.format(FLAGS.max_depth))
  # summary_merged = tf.summary.merge_all()
  # summary_writer = tf.summary.FileWriter(train_dir)

  # maze_queue = queue.Queue(100)

  # def maze_generator():
  #   maze_gen = MazeGenerator(
  #     height=FLAGS.maze_size,
  #     width=FLAGS.maze_size,
  #     density=FLAGS.maze_density)

  #   while True:
  #     maze_ims, maze_labels = maze_gen.generate_labelled_mazes(FLAGS.batch_size)
  #     maze_queue.put((maze_ims, maze_labels))

  # for thread_i in range(FLAGS.num_threads):
  #   t = threading.Thread(target=maze_generator)
  #   t.start()

  # for step in range(FLAGS.max_steps):
  #   start_time = time.time()
  #   maze_ims_np, maze_labels_np = maze_queue.get()

  #   _, loss_value, loss_preturns_val, loss_lambda_preturns_val, summary_str = model(maze_ims_np,maze_labels_np)
  #   duration = time.time() - start_time

  #   assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

  #   if step % 10 == 0:
  #     num_examples_per_step = FLAGS.batch_size
  #     examples_per_sec = num_examples_per_step / duration
  #     sec_per_batch = duration

  #     format_str = (
  #       '%s: step %d, loss = %.4f, loss_preturns = %.4f, loss_lambda_preturns = %.4f (%.1f examples/sec; %.3f '
  #       'sec/batch)')
  #     logger.info(format_str % (datetime.datetime.now(), step, loss_value, loss_preturns_val, loss_lambda_preturns_val,
  #                               examples_per_sec, sec_per_batch))

  #   if step % 100 == 0:
  #     summary_writer.add_summary(summary_str, step)

  #   # Save the model checkpoint periodically.
  #   if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
  #     checkpoint_path = os.path.join(train_dir, 'model.ckpt')
  #     saver.save(checkpoint_path, global_step=step)


def main(argv=None):
  train()

if __name__ == '__main__':
    app.run(main)