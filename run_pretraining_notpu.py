# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
# Copyright 2020 AMD MLSE Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pretraining BERT models"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import modeling
#import modeling_old as modeling
import optimization
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()

import time
from tensorflow.python.training.summary_io import SummaryWriterCache
from tensorflow.core.framework.summary_pb2 import Summary

from absl import flags, logging, app

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_integer(
    "num_report_steps", 10,
    "How frequently should summary information be reported and recorded.")

class LogSessionRunHook(tf.estimator.SessionRunHook):

    def __init__(self,
                 global_batch_size,
                 num_report_steps=10,
                 output_dir=None):
      self.global_batch_size = global_batch_size
      self.num_report_steps = num_report_steps
      self.output_dir=output_dir
      self.summary_writer=None

    def begin(self):
      if self.summary_writer is None and self.output_dir:
        self.summary_writer = SummaryWriterCache.get(self.output_dir)

    def after_create_session(self, session, coord):
      self.elapsed_secs = 0.
      self.count = 0

    def before_run(self, run_context):
      self.t0 = time.time()
      global_step = tf.compat.v1.train.get_global_step()
      fetches = [global_step, 'learning_rate:0', 'total_loss:0', 'mlm_loss:0', 'nsp_loss:0']
      return tf.estimator.SessionRunArgs(fetches=fetches)

    def _log_and_record(self, global_step, learning_rate, total_loss, mlm_loss, nsp_loss):
      time_per_step = self.elapsed_secs / self.count
      throughput = self.global_batch_size / time_per_step
      log_string = '  '
      log_string += 'Step = %6i'%(global_step)
      log_string += ', throughput = %6.1f'%(throughput)
      log_string += ', total_loss = %6.3f'%(total_loss)
      log_string += ', mlm_oss = %6.4e'%(mlm_loss)
      log_string += ', nsp_loss = %6.4e'%(nsp_loss)
      log_string += ', learning_rate = %6.4e'%(learning_rate)
      logging.info(log_string)

      if self.summary_writer is not None:
        throughput_summary = Summary(value=[Summary.Value(tag='throughput', simple_value=throughput)])
        self.summary_writer.add_summary(throughput_summary, global_step)
        total_loss_summary = Summary(value=[Summary.Value(tag='total_loss', simple_value=total_loss)])
        self.summary_writer.add_summary(total_loss_summary, global_step)

    def after_run(self, run_context, run_values):
      self.elapsed_secs += time.time() - self.t0
      self.count += 1
      global_step, learning_rate, total_loss, mlm_loss, nsp_loss = run_values.results[0:5]
      if (global_step % self.num_report_steps) == 0:
        self._log_and_record(global_step, learning_rate, total_loss, mlm_loss, nsp_loss)
        self.elapsed_secs = 0.
        self.count = 0


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, use_hvd, use_amp):
  """Returns `model_fn` closure for Estimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for Estimator."""

    logging.info("*** Features ***")
    for name in sorted(features.keys()):
      logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]
    next_sentence_labels = features["next_sentence_labels"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # model = modeling.BertModel(
    #     config=bert_config,
    #     is_training=is_training,
    #     input_ids=input_ids,
    #     input_mask=input_mask,
    #     token_type_ids=segment_ids,
    #     #use_one_hot_embeddings=use_one_hot_embeddings
    #     )
    model = modeling.BertModel(config=bert_config)
    model_output = model([input_ids, input_mask, segment_ids])

    (masked_lm_loss,
     masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
         bert_config, model.get_sequence_output(), model.get_embedding_table(),
         masked_lm_positions, masked_lm_ids, masked_lm_weights)

    (next_sentence_loss, next_sentence_example_loss,
     next_sentence_log_probs) = get_next_sentence_output(
         bert_config, model.get_pooled_output(), next_sentence_labels)

    total_loss = masked_lm_loss + next_sentence_loss

    masked_lm_loss = tf.identity(masked_lm_loss, name='mlm_loss')
    next_sentence_loss = tf.identity(next_sentence_loss, name='nsp_loss')
    total_loss = tf.identity(total_loss, name='total_loss')

    # TODO: Convert model loading to TF2
    tvars = tf.compat.v1.trainable_variables()

    initialized_variable_names = {}
    if init_checkpoint and (hvd == None or hvd.rank() == 0):
      logging.info("**** Init Checkpoint {} {} ****".format(hvd.rank(), init_checkpoint))
      (assignment_map, initialized_variable_names) = \
        modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

    logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      # train_op = optimization.create_optimizer(
      #     total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu, use_hvd, FLAGS.optimizer_type, use_amp)
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu, use_hvd, 'adam', use_amp)

      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
      def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                    masked_lm_weights, next_sentence_example_loss,
                    next_sentence_log_probs, next_sentence_labels):
        """Computes the loss and accuracy of the model."""
        masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                         [-1, masked_lm_log_probs.shape[-1]])
        masked_lm_predictions = tf.argmax(
            input=masked_lm_log_probs, axis=-1, output_type=tf.int32)
        masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
        masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
        masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
        # masked_lm_accuracy = tf.compat.v1.metrics.accuracy(
        #     labels=masked_lm_ids,
        #     predictions=masked_lm_predictions,
        #     weights=masked_lm_weights)
        masked_lm_accuracy = tf.keras.metrics.Accuracy()
        masked_lm_accuracy.update_state(masked_lm_ids, masked_lm_predictions, masked_lm_weights)
        # masked_lm_mean_loss = tf.compat.v1.metrics.mean(
        #     values=masked_lm_example_loss, weights=masked_lm_weights)
        masked_lm_mean_loss = tf.keras.metrics.Mean()
        masked_lm_mean_loss.update_state(masked_lm_example_loss, masked_lm_weights)

        next_sentence_log_probs = tf.reshape(
            next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
        next_sentence_predictions = tf.argmax(
            input=next_sentence_log_probs, axis=-1, output_type=tf.int32)
        next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
        # next_sentence_accuracy = tf.compat.v1.metrics.accuracy(
        #     labels=next_sentence_labels, predictions=next_sentence_predictions)
        next_sentence_accuracy = tf.keras.metrics.Accuracy()
        next_sentence_accuracy.update_state(next_sentence_labels, next_sentence_predictions)
        # next_sentence_mean_loss = tf.compat.v1.metrics.mean(
        #     values=next_sentence_example_loss)
        next_sentence_mean_loss = tf.keras.metrics.Mean()
        next_sentence_mean_loss.update_state(next_sentence_example_loss)

        return {
            "masked_lm_accuracy": masked_lm_accuracy,
            "masked_lm_loss": masked_lm_mean_loss,
            "next_sentence_accuracy": next_sentence_accuracy,
            "next_sentence_loss": next_sentence_mean_loss,
        }

      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metric_ops=metric_fn(
            masked_lm_example_loss=masked_lm_example_loss,
            masked_lm_log_probs=masked_lm_log_probs,
            masked_lm_ids=masked_lm_ids,
            masked_lm_weights=masked_lm_weights,
            next_sentence_example_loss=next_sentence_example_loss,
            next_sentence_log_probs=next_sentence_log_probs,
            next_sentence_labels=next_sentence_labels
          ))
    else:
      raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

    return output_spec

  return model_fn

# TODO: Change this lm part to Keras layers
def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.compat.v1.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.compat.v1.variable_scope("transform"):
      input_tensor = tf.compat.v1.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.compat.v1.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.compat.v1.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(input_tensor=log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(input_tensor=label_weights * per_example_loss)
    denominator = tf.reduce_sum(input_tensor=label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.compat.v1.variable_scope("cls/seq_relationship"):
    output_weights = tf.compat.v1.get_variable(
        "output_weights",
        shape=[2, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.compat.v1.get_variable(
        "output_bias", shape=[2], initializer=tf.compat.v1.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    # seq_rel_dense = tf.keras.layers.Dense(
    #   2,
    #   kernel_initializer=modeling.create_initializer(bert_config.initializer_range),
    # )
    # logits = seq_rel_dense(input_tensor)

    log_probs = tf.nn.log_softmax(logits, axis=-1)
    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(input_tensor=one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(input_tensor=per_example_loss)
    return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    name_to_features = {
        "input_ids":
            tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask":
            tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
            tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels":
            tf.io.FixedLenFeature([1], tf.int64),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      d = d.repeat()
      d = d.shuffle(buffer_size=len(input_files))

      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_cpu_threads, len(input_files))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.apply(
          tf.data.experimental.parallel_interleave(
              lambda filename: tf.data.TFRecordDataset(filename, compression_type='GZIP'),
              sloppy=is_training,
              cycle_length=cycle_length))
      d = d.shuffle(buffer_size=100)
    else:
      d = tf.data.TFRecordDataset(input_files, compression_type='GZIP')
      # Since we evaluate for a fixed number of steps we don't want to encounter
      # out-of-range exceptions.
      d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))
    return d

  return input_fn


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.io.parse_single_example(serialized=record, features=name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.cast(t, dtype=tf.int32)
    example[name] = t

  return example


def main(_):
  logging.set_verbosity(logging.INFO)

  if not FLAGS.do_train and not FLAGS.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  tf.io.gfile.makedirs(FLAGS.output_dir)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.io.gfile.glob(input_pattern))

  logging.info("*** Input Files ***")
  for input_file in input_files:
    logging.info("  %s" % input_file)

  run_config = tf.estimator.RunConfig(
    tf_random_seed=54321,
    model_dir=FLAGS.output_dir,
    save_checkpoints_steps=FLAGS.save_checkpoints_steps,
    log_step_count_steps=FLAGS.num_report_steps
  )

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps,
      use_tpu=False,
      use_one_hot_embeddings=False,
      use_hvd=False,
      use_amp=False)

  params = {
    'batch_size': FLAGS.train_batch_size
  }
  estimator = tf.compat.v1.estimator.Estimator(
      model_fn=model_fn,
      config=run_config,
      params=params)

  if FLAGS.do_train:
    logging.info("***** Running training *****")
    logging.info("  Batch size = %d", FLAGS.train_batch_size)
    train_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=True)

    hooks = []
    hooks.append(LogSessionRunHook(FLAGS.train_batch_size, FLAGS.num_report_steps, FLAGS.output_dir))

    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps, hooks=hooks)

  if FLAGS.do_eval:
    logging.info("***** Running evaluation *****")
    logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    eval_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=False)

    result = estimator.evaluate(
        input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.io.gfile.GFile(output_eval_file, "w") as writer:
      logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  app.run(main)
