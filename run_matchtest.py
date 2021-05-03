# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import modeling
import modeling_old
import optimization
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
tf.compat.v1.enable_eager_execution()

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


def copy_weight_from_old_to_new(model_old, model_new):
  # Copy embedding layer weights from old to new
  model_new._embedding_layer._word_embedding_table.assign(modeling_old.embedding_table.numpy())
  model_new._embedding_layer._token_type_table.assign(modeling_old.token_type_table.numpy())
  model_new._embedding_layer._position_embedding_table.assign(modeling_old.full_position_embeddings.numpy())

  # Loop through all the encoder layers
  for i in range(len(model_new._encoder_layers)):
    # Copy attention layer weights from old to new
    model_new._encoder_layers[i]._blocks['attention']._query_layer.kernel.assign(modeling_old.query_layer_objs[i].kernel.numpy())
    model_new._encoder_layers[i]._blocks['attention']._query_layer.bias.assign(modeling_old.query_layer_objs[i].bias.numpy())
    model_new._encoder_layers[i]._blocks['attention']._key_layer.kernel.assign(modeling_old.key_layer_objs[i].kernel.numpy())
    model_new._encoder_layers[i]._blocks['attention']._key_layer.bias.assign(modeling_old.key_layer_objs[i].bias.numpy())
    model_new._encoder_layers[i]._blocks['attention']._value_layer.kernel.assign(modeling_old.value_layer_objs[i].kernel.numpy())
    model_new._encoder_layers[i]._blocks['attention']._value_layer.bias.assign(modeling_old.value_layer_objs[i].bias.numpy())

    # Copy attention output weights from old to new
    model_new._encoder_layers[i]._blocks['attention_output'].kernel.assign(modeling_old.attention_output_objs[i].kernel.numpy())
    model_new._encoder_layers[i]._blocks['attention_output'].bias.assign(modeling_old.attention_output_objs[i].bias.numpy())

    # Copy ffn filer weights from old to new
    model_new._encoder_layers[i]._blocks['ffn_filter'].kernel.assign(modeling_old.intermediate_output_objs[i].kernel.numpy())
    model_new._encoder_layers[i]._blocks['ffn_filter'].bias.assign(modeling_old.intermediate_output_objs[i].bias.numpy())

    # Copy ffn output weights from old to new
    model_new._encoder_layers[i]._blocks['ffn_output'].kernel.assign(modeling_old.output_objs[i].kernel.numpy())
    model_new._encoder_layers[i]._blocks['ffn_output'].bias.assign(modeling_old.output_objs[i].bias.numpy())

  # Copy pooler weights from old to new
  model_new._pooler_layer._pooler_dense.kernel.assign(modeling_old.pooler_obj.kernel.numpy())
  model_new._pooler_layer._pooler_dense.bias.assign(modeling_old.pooler_obj.bias.numpy())


def main(_):
  logging.set_verbosity(logging.INFO)

  logging.info("*** Input Files ***")
  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.io.gfile.glob(input_pattern))
  for input_file in input_files:
    logging.info("  %s" % input_file)

  logging.info("*** Loading the dataset ***")

  # This is the name to feature mapping
  # name_to_features = {
  #   "input_ids":
  #       tf.io.FixedLenFeature([max_seq_length], tf.int64),
  #   "input_mask":
  #       tf.io.FixedLenFeature([max_seq_length], tf.int64),
  #   "segment_ids":
  #       tf.io.FixedLenFeature([max_seq_length], tf.int64),
  #   "masked_lm_positions":
  #       tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
  #   "masked_lm_ids":
  #       tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
  #   "masked_lm_weights":
  #       tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
  #   "next_sentence_labels":
  #       tf.io.FixedLenFeature([1], tf.int64),
  # }

  raw_dataset = tf.data.TFRecordDataset(input_files, compression_type='GZIP')
  for raw_record in raw_dataset.take(1):
    # Parse one example
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())

    # Extract the features from the example, and convert them to tensor
    features = example.features.feature
    input_ids = tf.convert_to_tensor(example.features.feature['input_ids'].int64_list.value, name='input_ids')
    input_mask = tf.convert_to_tensor(example.features.feature['input_mask'].int64_list.value, name='input_mask')
    segment_ids = tf.convert_to_tensor(example.features.feature['segment_ids'].int64_list.value, name='segment_ids')

    # Expand the batch dimension (we use batch_size 1)
    input_ids = tf.expand_dims(input_ids, axis=0)
    input_mask = tf.expand_dims(input_mask, axis=0)
    segment_ids = tf.expand_dims(segment_ids, axis=0)
    print('input_ids shape:', input_ids.shape.as_list())
    print('input_mask shape:', input_mask.shape.as_list())
    print('segment_ids shape:', segment_ids.shape.as_list())

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    is_training = False


    model_old = modeling_old.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False)
    output_old = model_old.get_pooled_output()

    model_new = modeling.BertModel(
        config=bert_config)
    # Run it first to build the model
    output_new = model_new([input_ids, input_mask, segment_ids])

    # Compare. They should not match.
    std = tf.math.reduce_std(output_new - output_old)
    print("Standard Deviation (without sync weight):", std)

    # Copy weight and then match again
    copy_weight_from_old_to_new(model_old, model_new)
    print("Weight copied from old model to new model.")

    # Try it again
    output_new = model_new([input_ids, input_mask, segment_ids])
    std = tf.math.reduce_std(output_new - output_old)
    print("Standard Deviation (with sync weight):", std)

    # for w in model_new.trainable_weights:
    #   print(w.name, w.shape)
    # tvar = tf.compat.v1.trainable_variables()
    # print(tvar)

    model_new.save_weights("./ckpts/weights_ckpt")
    model_new.save("./model/my_model")

    model_new2 = modeling.BertModel(
        config=bert_config)

    output_new2 = model_new2([input_ids, input_mask, segment_ids])
    std = tf.math.reduce_std(output_new2 - output_old)
    print("Standard Deviation (without loading weight):", std)

    print("Load from saved weight")
    model_new2.load_weights("./ckpts/weights_ckpt")
    output_new2 = model_new2([input_ids, input_mask, segment_ids])
    std = tf.math.reduce_std(output_new2 - output_old)
    print("Standard Deviation (with loaded weight):", std)
    
    #model_new3 = tf.keras.models.load_model("./model/my_model")

    #test = tf.compat.v1.get_default_graph().get_tensor_by_name("bert/embeddings/word_embeddings:0")
    # graph = tf.compat.v1.get_default_graph()

    # print(graph.collections)
    # print(graph.get_collection('variables'))
    # print(graph.get_collection('local_variables'))
    # print(graph.get_collection('trainable_variables'))
    # test = tf.compat.v1.get_default_graph().collections['variables']
    # print(test)

  return


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  app.run(main)
