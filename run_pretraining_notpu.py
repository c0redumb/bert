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
import optimization
import tensorflow as tf

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


def model_fn_builder():
  """Returns `model_fn` closure for Estimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for Estimator."""

    ### Input Features ###
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

    ### Model ###
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    t_model = get_pretraining_model(bert_config, params)
    [masked_lm_log_probs, next_sentence_log_probs] = t_model([input_ids, input_mask, segment_ids, masked_lm_positions])

    # print(t_model.summary())
    tvars = t_model.trainable_variables
    logging.info("**** Trainable Variables ****")
    for var in tvars:
      logging.info("  name = %s, shape = %s", var.name, var.shape)

    masked_lm_loss = PredictMaskedLM.get_loss(
                                              log_probs=masked_lm_log_probs,
                                              bert_config=bert_config,
                                              masked_lm_positions=masked_lm_positions,
                                              masked_lm_ids=masked_lm_ids,
                                              masked_lm_weights=masked_lm_weights)
    masked_lm_metrics = PredictMaskedLM.get_metrics(
                                              log_probs=masked_lm_log_probs,
                                              per_example_loss=masked_lm_loss["per_example_loss"],
                                              masked_lm_ids=masked_lm_ids,
                                              masked_lm_weights=masked_lm_weights)

    next_sentence_loss = PredictNextSentence.get_loss(
                                              log_probs=next_sentence_log_probs,
                                              next_sentence_labels=next_sentence_labels)
    next_sentence_metrics = PredictNextSentence.get_metrics(
                                              log_probs=next_sentence_log_probs, 
                                              per_example_loss = next_sentence_loss["per_example_loss"],
                                              next_sentence_labels=next_sentence_labels)

    total_loss = masked_lm_loss["loss"] + next_sentence_loss["loss"]

    masked_lm_loss = tf.identity(masked_lm_loss["loss"], name='mlm_loss')
    next_sentence_loss = tf.identity(next_sentence_loss["loss"], name='nsp_loss')
    total_loss = tf.identity(total_loss, name='total_loss')



    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      # train_op = optimization.create_optimizer(
      #     total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu, use_hvd, FLAGS.optimizer_type, use_amp)
      train_op = optimization.create_optimizer(total_loss,
                                              params["learning_rate"],
                                              params["num_train_steps"],
                                              params["num_warmup_steps"],
                                              None, None, "adam", None)

      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metric_ops={**masked_lm_metrics, **next_sentence_metrics},
          )
    else:
      raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def get_pretraining_model(bert_config, params):
  """Build model for BERT pretraining"""
  max_seq_length = params["max_seq_length"]
  max_predictions_per_seq = params["max_predictions_per_seq"]

  # These are the input tensors to the model
  # TODO: Change int32 to int64, because we don't care about TPU
  input_ids = tf.keras.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_ids")
  input_mask = tf.keras.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
  segment_ids = tf.keras.Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids")
  masked_lm_positions = tf.keras.Input(shape=(max_predictions_per_seq,), dtype=tf.int32, name="mlm_positions")

  # BERT model
  bert = modeling.BertModel(config=bert_config)
  bert_pooled_output = bert([input_ids, input_mask, segment_ids])
  bert_sequence_output = bert.get_sequence_output()
  bert_embedding_weight = bert.get_embedding_table()

  # Masked LM prediction
  pred_masked_lm = PredictMaskedLM(
                                  bert_config=bert_config,
                                  embedding_weight=bert_embedding_weight)
  masked_lm_log_probs = pred_masked_lm(
                                  bert_sequence_output=bert_sequence_output, 
                                  masked_lm_positions=masked_lm_positions)

  # Next Sentence prediction
  pred_next_sentence = PredictNextSentence(bert_config=bert_config)
  next_sentence_log_probs = pred_next_sentence(bert_pooled_output=bert_pooled_output)

  training_model = tf.keras.Model(
    inputs=[input_ids, input_mask, segment_ids, masked_lm_positions],
    outputs = [masked_lm_log_probs, next_sentence_log_probs])

  return training_model


class PredictMaskedLM(tf.keras.layers.Layer):
    """Masked LM Prediction for Pretraining"""
    def __init__(self,
                bert_config,
                embedding_weight,
                 **kwargs):
      self._bert_config = bert_config
      self._embedding_weight = embedding_weight
      super(PredictMaskedLM, self).__init__(**kwargs)

    def build(self, input_shape):
      self._nonlinear = tf.keras.layers.Dense(
              units=self._bert_config.hidden_size,
              activation=modeling.get_activation(self._bert_config.hidden_act),
              kernel_initializer=modeling.create_initializer(self._bert_config.initializer_range),
              name='masked_lm_dense'
      )
      self._layernorm = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-12,
            name="masked_lm_layernorm")
      self._bias = self.add_weight(
              'masked_lm_bias',
              shape=[self._bert_config.vocab_size,],
              initializer=tf.keras.initializers.Zeros,
              trainable=True)
      super(PredictMaskedLM, self).build(input_shape)

    def call(self, bert_sequence_output, masked_lm_positions):
      x = PredictMaskedLM._gather_indexes(bert_sequence_output, masked_lm_positions)
      x = self._nonlinear(x)
      x = self._layernorm(x)
      logits = tf.matmul(x, self._embedding_weight, transpose_b=True)
      logits = tf.nn.bias_add(logits, self._bias)
      log_probs = tf.nn.log_softmax(logits, axis=-1)
      return log_probs

    @staticmethod
    def get_loss(log_probs, bert_config, masked_lm_positions, masked_lm_ids, masked_lm_weights):
      label_ids = tf.reshape(masked_lm_ids, [-1])
      label_weights = tf.reshape(masked_lm_weights, [-1])

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

      return {"loss": loss, "per_example_loss": per_example_loss}

    @staticmethod
    def get_metrics(log_probs, per_example_loss, masked_lm_ids, masked_lm_weights):
      label_ids = tf.reshape(masked_lm_ids, [-1])
      label_weights = tf.reshape(masked_lm_weights, [-1])

      log_probs = tf.reshape(log_probs, [-1, log_probs.shape[-1]])
      predictions = tf.argmax(input=log_probs, axis=-1, output_type=tf.int32)
      per_example_loss = tf.reshape(per_example_loss, [-1])

      accuracy = tf.keras.metrics.Accuracy()
      accuracy.update_state(label_ids, predictions, label_weights)
      mean_loss = tf.keras.metrics.Mean()
      mean_loss.update_state(per_example_loss, label_weights)

      return {"masked_lm_accuracy": accuracy, "masked_lm_loss": mean_loss}

    @staticmethod
    def _gather_indexes(sequence_tensor, positions):
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


class PredictNextSentence(tf.keras.layers.Layer):
    """Next Sentence Prediction for Pretraining"""
    def __init__(self,
                 bert_config,
                 **kwargs):
      self._bert_config = bert_config
      super(PredictNextSentence, self).__init__(**kwargs)

    def build(self, input_shape):
      self._linear = tf.keras.layers.Dense(
              units=2,
              #activation=modeling.get_activation(self._bert_config.hidden_act),
              kernel_initializer=modeling.create_initializer(self._bert_config.initializer_range),
              name='next_sentence_dense'
      )
      super(PredictNextSentence, self).build(input_shape)

    def call(self, bert_pooled_output):
      logits = self._linear(bert_pooled_output)
      log_probs = tf.nn.log_softmax(logits, axis=-1)
      return log_probs

    @staticmethod
    def get_loss(log_probs, next_sentence_labels):
      labels = tf.reshape(next_sentence_labels, [-1])
      one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
      per_example_loss = -tf.reduce_sum(input_tensor=one_hot_labels * log_probs, axis=-1)
      loss = tf.reduce_mean(input_tensor=per_example_loss)

      return {"loss": loss, "per_example_loss": per_example_loss}

    @staticmethod
    def get_metrics(log_probs, per_example_loss, next_sentence_labels):
      labels = tf.reshape(next_sentence_labels, [-1])

      log_probs = tf.reshape(log_probs, [-1, log_probs.shape[-1]])
      predictions = tf.argmax(input=log_probs, axis=-1, output_type=tf.int32)

      accuracy = tf.keras.metrics.Accuracy()
      accuracy.update_state(labels, predictions)
      mean_loss = tf.keras.metrics.Mean()
      mean_loss.update_state(per_example_loss)

      return {"next_sentense_accuracy": accuracy, "next_sentense_loss": mean_loss}


def input_fn_builder(input_files,
                     batch_size,
                     is_training,
                     num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to Estimator."""

  def input_fn(params):
    """The actual input function."""
    max_seq_length = params["max_seq_length"]
    max_predictions_per_seq = params["max_predictions_per_seq"]

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

  model_fn = model_fn_builder()

  params = {
    "bert_config_file": FLAGS.bert_config_file,
    "max_seq_length": FLAGS.max_seq_length,
    "max_predictions_per_seq": FLAGS.max_predictions_per_seq,
    "learning_rate": FLAGS.learning_rate,
    "num_train_steps": FLAGS.num_train_steps,
    "num_warmup_steps": FLAGS.num_warmup_steps
  }

  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config,
      params=params,
      warm_start_from=FLAGS.init_checkpoint)

  if FLAGS.do_train:
    logging.info("***** Running training *****")
    logging.info("  Training Batch size = %d", FLAGS.train_batch_size)
    train_input_fn = input_fn_builder(
        input_files=input_files,
        batch_size=FLAGS.train_batch_size,
        is_training=True)

    hooks = []
    hooks.append(LogSessionRunHook(FLAGS.train_batch_size, FLAGS.num_report_steps, FLAGS.output_dir))

    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps, hooks=hooks)

  if FLAGS.do_eval:
    logging.info("***** Running evaluation *****")
    logging.info("  Evaluation Batch size = %d", FLAGS.eval_batch_size)

    eval_input_fn = input_fn_builder(
        input_files=input_files,
        batch_size=FLAGS.eval_batch_size,
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
