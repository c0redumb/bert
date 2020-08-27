# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""The main BERT model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


class BertConfig(object):
    """Configuration for `BertModel`."""

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
          vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
          hidden_size: Size of the encoder layers and the pooler layer.
          num_hidden_layers: Number of hidden layers in the Transformer encoder.
          num_attention_heads: Number of attention heads for each attention layer in
            the Transformer encoder.
          intermediate_size: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
          hidden_act: The non-linear activation function (function or string) in the
            encoder and pooler.
          hidden_dropout_prob: The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler.
          attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.
          max_position_embeddings: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
          type_vocab_size: The vocabulary size of the `token_type_ids` passed into
            `BertModel`.
          initializer_range: The stdev of the truncated_normal_initializer for
            initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.io.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertModel(object):
    """BERT model ("Bidirectional Encoder Representations from Transformers").

    Example usage:

    ```python
    # Already been converted into WordPiece token ids
    input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
    input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
    token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
      num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config, is_training=True,
      input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

    label_embeddings = tf.get_variable(...)
    pooled_output = model.get_pooled_output()
    logits = tf.matmul(pooled_output, label_embeddings)
    ...
    ```
    """

    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 use_one_hot_embeddings=False,
                 scope=None):
        """Constructor for BertModel.

        Args:
          config: `BertConfig` instance.
          is_training: bool. true for training model, false for eval model. Controls
            whether dropout will be applied.
          input_ids: int32 Tensor of shape [batch_size, seq_length].
          input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
          token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
          use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
            embeddings or tf.embedding_lookup() for the word embeddings.
          scope: (optional) variable scope. Defaults to "bert".

        Raises:
          ValueError: The config is invalid or one of the input tensor shapes
            is invalid.
        """
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        input_shape = get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        # print(input_shape)

        if input_mask is None:
            input_mask = tf.ones(
                shape=[batch_size, seq_length], dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.zeros(
                shape=[batch_size, seq_length], dtype=tf.int32)

        with tf.compat.v1.variable_scope(scope, default_name="bert"):
            # with tf.compat.v1.variable_scope("embeddings"):
            #     # Perform embedding lookup on the word ids.
            #     (self.embedding_output, self.embedding_table) = embedding_lookup(
            #         input_ids=input_ids,
            #         vocab_size=config.vocab_size,
            #         embedding_size=config.hidden_size,
            #         initializer_range=config.initializer_range,
            #         word_embedding_name="word_embeddings",
            #         use_one_hot_embeddings=use_one_hot_embeddings)

            #     # Add positional embeddings and token type embeddings, then layer
            #     # normalize and perform dropout.
            #     self.embedding_output = embedding_postprocessor(
            #         input_tensor=self.embedding_output,
            #         use_token_type=True,
            #         token_type_ids=token_type_ids,
            #         token_type_vocab_size=config.type_vocab_size,
            #         token_type_embedding_name="token_type_embeddings",
            #         use_position_embeddings=True,
            #         position_embedding_name="position_embeddings",
            #         initializer_range=config.initializer_range,
            #         max_position_embeddings=config.max_position_embeddings,
            #         dropout_prob=config.hidden_dropout_prob)
            (self.embedding_output, self.embedding_table) = Embedding(
                embedding_size=config.hidden_size,
                vocab_size=config.vocab_size,
                token_type_vocab_size=config.type_vocab_size,
                max_position_embeddings=config.max_position_embeddings,
                init_range=config.initializer_range,
                dropout_prob=config.hidden_dropout_prob,
                name="embedding_layer"
            )(input_ids, token_type_ids, training=is_training)

            # with tf.compat.v1.variable_scope("encoder"):
            # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
            # mask of shape [batch_size, seq_length, seq_length] which is used
            # for the attention scores.
            attention_mask = create_attention_mask_from_input_mask(
                input_ids, input_mask)

            # Run the stacked transformer.
            # `sequence_output` shape = [batch_size, seq_length, hidden_size].
            # self.all_encoder_layers = transformer_model(
            #     input_tensor=self.embedding_output,
            #     attention_mask=attention_mask,
            #     hidden_size=config.hidden_size,
            #     num_hidden_layers=config.num_hidden_layers,
            #     num_attention_heads=config.num_attention_heads,
            #     intermediate_size=config.intermediate_size,
            #     intermediate_act_fn=get_activation(config.hidden_act),
            #     hidden_dropout_prob=config.hidden_dropout_prob,
            #     attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            #     initializer_range=config.initializer_range,
            #     do_return_all_layers=True,
            #     training=is_training)
            self.all_encoder_layers = []
            layer_input = self.embedding_output

            for idx in range(config.num_hidden_layers):
                layer_output = TransformerEncoder(
                    num_attention_heads=config.num_attention_heads,
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    intermediate_act_fn=config.hidden_act,
                    attn_probs_dropout_prob=config.attention_probs_dropout_prob,
                    other_dropout_prob=config.hidden_dropout_prob,
                    init_range=config.initializer_range,
                    name="encoder_layer_%d" % idx
                )(layer_input, attention_mask=attention_mask, training=is_training)

                self.all_encoder_layers.append(layer_output)
                layer_input = layer_output

            self.sequence_output = self.all_encoder_layers[-1]
            # The "pooler" converts the encoded sequence tensor of shape
            # [batch_size, seq_length, hidden_size] to a tensor of shape
            # [batch_size, hidden_size]. This is necessary for segment-level
            # (or segment-pair-level) classification tasks where we need a fixed
            # dimensional representation of the segment.
            # with tf.compat.v1.variable_scope("pooler"):
            #     # We "pool" the model by simply taking the hidden state corresponding
            #     # to the first token. We assume that this has been pre-trained
            #     first_token_tensor = tf.squeeze(
            #         self.sequence_output[:, 0:1, :], axis=1)
            #     self.pooled_output = tf.compat.v1.layers.dense(
            #         first_token_tensor,
            #         config.hidden_size,
            #         activation=tf.tanh,
            #         kernel_initializer=create_initializer(config.initializer_range))
            self.pooled_output = Pooler(
                hidden_size=config.hidden_size,
                act_fn="tanh",
                init_range=config.initializer_range,
                name="pooler_layer"
            )(self.sequence_output)

    def get_pooled_output(self):
        return self.pooled_output

    def get_sequence_output(self):
        """Gets final hidden layer of encoder.

        Returns:
          float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
          to the final hidden of the transformer encoder.
        """
        return self.sequence_output

    def get_all_encoder_layers(self):
        return self.all_encoder_layers

    def get_embedding_output(self):
        """Gets output of the embedding lookup (i.e., input to the transformer).

        Returns:
          float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
          to the output of the embedding layer, after summing the word
          embeddings with the positional embeddings and the token type embeddings,
          then performing layer normalization. This is the input to the transformer.
        """
        return self.embedding_output

    def get_embedding_table(self):
        return self.embedding_table


def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      x: float Tensor to perform activation.

    Returns:
      `x` with the GELU activation applied.
    """
    try:
        return tf.nn.gelu(x)
    except:
        cdf = 0.5 * (1.0 + tf.tanh(
            (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
        return x * cdf


def get_activation(activation_string):
    """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

    Args:
      activation_string: String name of the activation function.

    Returns:
      A Python function corresponding to the activation function. If
      `activation_string` is None, empty, or "linear", this will return None.
      If `activation_string` is not a string, it will return `activation_string`.

    Raises:
      ValueError: The `activation_string` does not correspond to a known
        activation.
    """

    # We assume that anything that"s not a string is already an activation
    # function, so we just return it.
    if not isinstance(activation_string, six.string_types):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return tf.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
      (name, var) = (x[0], x[1])
      if name not in name_to_variable:
        continue
      #assignment_map[name] = name
      assignment_map[name] = name_to_variable[name]
      initialized_variable_names[name] = 1
      initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


def dropout(input_tensor, dropout_prob):
    """Perform dropout.

    Args:
      input_tensor: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1 - (1.0 - dropout_prob))
    return output


def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    # return tf.contrib.layers.layer_norm(
    #     inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)
    return tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12)(inputs=input_tensor)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
    """Runs layer normalization followed by dropout."""
    output_tensor = layer_norm(input_tensor, name)
    output_tensor = dropout(output_tensor, dropout_prob)
    return output_tensor


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    # return tf.compat.v1.truncated_normal_initializer(stddev=initializer_range)
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


class Embedding(tf.keras.layers.Layer):
    """Embedding layer in BERT

    This layer performs word embeddings, segment type embedding, and
    position embeddings as described in the BERT paper.
    """

    def __init__(self,
                 embedding_size,
                 vocab_size,
                 token_type_vocab_size=None,
                 max_position_embeddings=None,
                 init_range=0.0,
                 dropout_prob=0.0,
                 **kwargs):
        """Initialize Embedding for BERT

        Args:
          embedding_size: int, size of embedding, matchs hidden size of encoder
          vocab_size: int, size of the vocabulary
          token_type_vocab_size: int, size of token types (for next sentense prediction)
          max_position_embeddings: int, max position embedding range
          init_range: float, range of initializer
          dropout_prob: float, dropout rate for training
        """
        super(Embedding, self).__init__(**kwargs)
        self._embedding_size = embedding_size
        self._vocab_size = vocab_size
        self._token_type_vocab_size = token_type_vocab_size
        self._max_position_embeddings = max_position_embeddings
        self._init_range = init_range
        self._dropout_prob = dropout_prob

    def build(self, input_shape):
        self._word_embedding_table = self.add_weight(
            shape=[self._vocab_size, self._embedding_size],
            initializer=create_initializer(self._init_range),
            trainable=True,
            name="word_embedding")
        self._token_type_table = self.add_weight(
            shape=[self._token_type_vocab_size, self._embedding_size],
            initializer=create_initializer(self._init_range),
            trainable=True,
            name="type_embedding")
        self._position_embedding_table = self.add_weight(
            shape=[self._max_position_embeddings, self._embedding_size],
            initializer=create_initializer(self._init_range),
            trainable=True,
            name="position_embedding")
        self._layer_norm = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-12,
            name="embedding_norm")
        self._dropout = tf.keras.layers.Dropout(
            rate=self._dropout_prob,
            name="embedding_dropout")
        super(Embedding, self).build(input_shape)

    def get_config(self):
        return {
            "embedding_size": self._embedding_size,
            "vocab_size": self._vocab_size,
            "token_type_vocab_size": self._token_type_vocab_size,
            "max_position_embeddings": self._max_position_embeddings,
            "init_range": self._init_range,
            "dropout_prob": self._dropout_prob,
        }

    def call(self, input_ids, token_type_ids, training=None):
        """Return outputs of the embedding

        Args:
          input_ids: input sequence with shape [batch_size, seq_length]
          token_type_ids: input type sequence with shape [batch_size, seq_length]
          training: boolean, whether in training mode or not

        Returns:
          Output of the embedding for BERT
          tensor with shape [batch_size, seq_length, embedding_size]
        """

        # Retrieve dynamically known shapes
        input_shape = input_ids.shape.as_list()
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        output_shape = input_shape.copy()
        output_shape.append(self._embedding_size)
        #print(input_shape, output_shape)

        # Word Embedding
        flat_input_ids = tf.reshape(input_ids, [-1])
        word_embedding_output = tf.gather(
            self._word_embedding_table, flat_input_ids)
        word_embedding_output = tf.reshape(word_embedding_output, output_shape)

        # Token type embedding
        # This vocab will be small so we always do one-hot here, since it is always
        # faster for a small vocabulary.
        flat_token_type_ids = tf.reshape(token_type_ids, [-1])
        one_hot_token_type_ids = tf.one_hot(
            flat_token_type_ids, depth=self._token_type_vocab_size)
        token_type_embeddings_output = tf.matmul(
            one_hot_token_type_ids, self._token_type_table)
        token_type_embeddings_output = tf.reshape(
            token_type_embeddings_output, output_shape)

        # Position embedding
        assert_op = tf.compat.v1.assert_less_equal(
            seq_length, self._max_position_embeddings)
        with tf.control_dependencies([assert_op]):
            # Since the position embedding table is a learned variable, we create it
            # using a (long) sequence length `max_position_embeddings`. The actual
            # sequence length might be shorter than this, for faster training of
            # tasks that do not have long sequences.
            #
            # So `full_position_embeddings` is effectively an embedding table
            # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
            # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
            # perform a slice.
            position_embeddings_output = tf.slice(self._position_embedding_table, [0, 0],
                                                  [seq_length, -1])
            # Now this has dimension [seq_length, embedding_size], we just need
            # to extend one dimension so it is broadcasted to the entire batch.
            position_embeddings_output = tf.expand_dims(
                position_embeddings_output, axis=0)

        # Summing all the embeddings together
        embedding_output = word_embedding_output + \
            token_type_embeddings_output + position_embeddings_output

        # Layer Norm and Drop out
        embedding_output = self._layer_norm(
            embedding_output, training=training)
        embedding_output = self._dropout(embedding_output, training=training)

        return (embedding_output, self._word_embedding_table)

    def get_embedding_table(self):
        return self._word_embedding_table


class Pooler(tf.keras.layers.Layer):
    """Pooler layer in BERT

    This is just a simple dense layer on the output of the first state
    """

    def __init__(self,
                 hidden_size,
                 act_fn="tanh",
                 init_range=0.0,
                 **kwargs):
        """Initialize Pooler for BERT

        Args:
          hidden_size: int, size of dense layer
          act_fn: activation function
          init_range: float, range of initializer
        """
        super(Pooler, self).__init__(**kwargs)
        self._hidden_size = hidden_size
        self._act_fn = act_fn
        self._init_range = init_range

    def build(self, input_shape):
        self._pooler_dense = tf.keras.layers.Dense(
            self._hidden_size,
            activation=get_activation(self._act_fn),
            kernel_initializer=create_initializer(self._init_range),
            name="pooler_dense")
        super(Pooler, self).build(input_shape)

    def get_config(self):
        return {
            "hidden_size": self._hidden_size,
            "act_fn": self._act_fn,
            "init_range": self._init_range
        }

    def call(self, input_tensor, training=None):
        """Return outputs of the embedding

        Args:
          input_tensor: input sequence with shape [batch_size, seq_length, hidden_size]
          training: boolean, whether in training mode or not

        Returns:
          Output of the pooler for BERT
          tensor with shape [batch_size, embedding_size]
        """

        # Retrieve dynamically known shapes
        # input_shape = input_tensor.shape.as_list()
        # batch_size = input_shape[0]
        # seq_length = input_shape[1]

        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        first_token_tensor = tf.squeeze(input_tensor[:, 0:1, :], axis=1)
        output = self._pooler_dense(first_token_tensor)

        return output


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False):
    """Looks up words embeddings for id tensor.

    Args:
      input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
        ids.
      vocab_size: int. Size of the embedding vocabulary.
      embedding_size: int. Width of the word embeddings.
      initializer_range: float. Embedding initialization range.
      word_embedding_name: string. Name of the embedding table.
      use_one_hot_embeddings: bool. If True, use one-hot method for word
        embeddings. If False, use `tf.gather()`.

    Returns:
      float Tensor of shape [batch_size, seq_length, embedding_size].
    """
    # This function assumes that the input is of shape [batch_size, seq_length,
    # num_inputs].
    #
    # If the input is a 2D tensor of shape [batch_size, seq_length], we
    # reshape to [batch_size, seq_length, 1].
    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=[-1])

    embedding_table = tf.compat.v1.get_variable(
        name=word_embedding_name,
        shape=[vocab_size, embedding_size],
        initializer=create_initializer(initializer_range))

    flat_input_ids = tf.reshape(input_ids, [-1])
    if use_one_hot_embeddings:
        one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
        output = tf.matmul(one_hot_input_ids, embedding_table)
    else:
        output = tf.gather(embedding_table, flat_input_ids)

    input_shape = get_shape_list(input_ids)

    output = tf.reshape(output,
                        input_shape[0:-1] + [input_shape[-1] * embedding_size])
    return (output, embedding_table)


def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
    """Performs various post-processing on a word embedding tensor.

    Args:
      input_tensor: float Tensor of shape [batch_size, seq_length,
        embedding_size].
      use_token_type: bool. Whether to add embeddings for `token_type_ids`.
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
        Must be specified if `use_token_type` is True.
      token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
      token_type_embedding_name: string. The name of the embedding table variable
        for token type ids.
      use_position_embeddings: bool. Whether to add position embeddings for the
        position of each token in the sequence.
      position_embedding_name: string. The name of the embedding table variable
        for positional embeddings.
      initializer_range: float. Range of the weight initialization.
      max_position_embeddings: int. Maximum sequence length that might ever be
        used with this model. This can be longer than the sequence length of
        input_tensor, but cannot be shorter.
      dropout_prob: float. Dropout probability applied to the final output tensor.

    Returns:
      float tensor with same shape as `input_tensor`.

    Raises:
      ValueError: One of the tensor shapes or input values is invalid.
    """
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    output = input_tensor

    if use_token_type:
        if token_type_ids is None:
            raise ValueError("`token_type_ids` must be specified if"
                             "`use_token_type` is True.")
        token_type_table = tf.compat.v1.get_variable(
            name=token_type_embedding_name,
            shape=[token_type_vocab_size, width],
            initializer=create_initializer(initializer_range))
        # This vocab will be small so we always do one-hot here, since it is always
        # faster for a small vocabulary.
        flat_token_type_ids = tf.reshape(token_type_ids, [-1])
        one_hot_ids = tf.one_hot(
            flat_token_type_ids, depth=token_type_vocab_size)
        token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
        token_type_embeddings = tf.reshape(token_type_embeddings,
                                           [batch_size, seq_length, width])
        output += token_type_embeddings

    if use_position_embeddings:
        assert_op = tf.compat.v1.assert_less_equal(
            seq_length, max_position_embeddings)
        with tf.control_dependencies([assert_op]):
            full_position_embeddings = tf.compat.v1.get_variable(
                name=position_embedding_name,
                shape=[max_position_embeddings, width],
                initializer=create_initializer(initializer_range))
            # Since the position embedding table is a learned variable, we create it
            # using a (long) sequence length `max_position_embeddings`. The actual
            # sequence length might be shorter than this, for faster training of
            # tasks that do not have long sequences.
            #
            # So `full_position_embeddings` is effectively an embedding table
            # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
            # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
            # perform a slice.
            position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                           [seq_length, -1])
            num_dims = len(output.shape.as_list())

            # Only the last two dimensions are relevant (`seq_length` and `width`), so
            # we broadcast among the first dimensions, which is typically just
            # the batch size.
            position_broadcast_shape = []
            for _ in range(num_dims - 2):
                position_broadcast_shape.append(1)
            position_broadcast_shape.extend([seq_length, width])
            position_embeddings = tf.reshape(position_embeddings,
                                             position_broadcast_shape)
            output += position_embeddings

    output = layer_norm_and_dropout(output, dropout_prob)
    return output


def create_attention_mask_from_input_mask(from_tensor, to_mask):
    """Create 3D attention mask from a 2D tensor mask.

    Args:
      from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
      to_mask: int32 Tensor of shape [batch_size, to_seq_length].

    Returns:
      float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(
        tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    #
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.ones(
        shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask

    return mask


def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
    """Performs multi-headed attention from `from_tensor` to `to_tensor`.

    This is an implementation of multi-headed attention based on "Attention
    is all you Need". If `from_tensor` and `to_tensor` are the same, then
    this is self-attention. Each timestep in `from_tensor` attends to the
    corresponding sequence in `to_tensor`, and returns a fixed-with vector.

    This function first projects `from_tensor` into a "query" tensor and
    `to_tensor` into "key" and "value" tensors. These are (effectively) a list
    of tensors of length `num_attention_heads`, where each tensor is of shape
    [batch_size, seq_length, size_per_head].

    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor and returned.

    In practice, the multi-headed attention are done with transposes and
    reshapes rather than actual separate tensors.

    Args:
      from_tensor: float Tensor of shape [batch_size, from_seq_length,
        from_width].
      to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
      attention_mask: (optional) int32 Tensor of shape [batch_size,
        from_seq_length, to_seq_length]. The values should be 1 or 0. The
        attention scores will effectively be set to -infinity for any positions in
        the mask that are 0, and will be unchanged for positions that are 1.
      num_attention_heads: int. Number of attention heads.
      size_per_head: int. Size of each attention head.
      query_act: (optional) Activation function for the query transform.
      key_act: (optional) Activation function for the key transform.
      value_act: (optional) Activation function for the value transform.
      attention_probs_dropout_prob: (optional) float. Dropout probability of the
        attention probabilities.
      initializer_range: float. Range of the weight initializer.
      do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
        * from_seq_length, num_attention_heads * size_per_head]. If False, the
        output will be of shape [batch_size, from_seq_length, num_attention_heads
        * size_per_head].
      batch_size: (Optional) int. If the input is 2D, this might be the batch size
        of the 3D version of the `from_tensor` and `to_tensor`.
      from_seq_length: (Optional) If the input is 2D, this might be the seq length
        of the 3D version of the `from_tensor`.
      to_seq_length: (Optional) If the input is 2D, this might be the seq length
        of the 3D version of the `to_tensor`.

    Returns:
      float Tensor of shape [batch_size, from_seq_length,
        num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
        true, this will be of shape [batch_size * from_seq_length,
        num_attention_heads * size_per_head]).

    Raises:
      ValueError: Any of the arguments or tensor shapes are invalid.
    """

    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                             seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(a=output_tensor, perm=[0, 2, 1, 3])
        return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

    if len(from_shape) != len(to_shape):
        raise ValueError(
            "The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError(
                "When passing in rank 2 tensors to attention_layer, the values "
                "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                "must all be specified.")

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`

    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)

    # `query_layer` = [B*F, N*H]
    query_layer = tf.compat.v1.layers.dense(
        from_tensor_2d,
        num_attention_heads * size_per_head,
        activation=query_act,
        name="query",
        kernel_initializer=create_initializer(initializer_range))

    # `key_layer` = [B*T, N*H]
    key_layer = tf.compat.v1.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=key_act,
        name="key",
        kernel_initializer=create_initializer(initializer_range))

    # `value_layer` = [B*T, N*H]
    value_layer = tf.compat.v1.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=value_act,
        name="value",
        kernel_initializer=create_initializer(initializer_range))

    # `query_layer` = [B, N, F, H]
    query_layer = transpose_for_scores(query_layer, batch_size,
                                       num_attention_heads, from_seq_length,
                                       size_per_head)

    # `key_layer` = [B, N, T, H]
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                     to_seq_length, size_per_head)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    # `attention_scores` = [B, N, F, T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(size_per_head)))

    if attention_mask is not None:
        # `attention_mask` = [B, 1, F, T]
        attention_mask = tf.expand_dims(attention_mask, axis=[1])

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_scores += adder

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

    # `value_layer` = [B, T, N, H]
    value_layer = tf.reshape(
        value_layer,
        [batch_size, to_seq_length, num_attention_heads, size_per_head])

    # `value_layer` = [B, N, T, H]
    value_layer = tf.transpose(a=value_layer, perm=[0, 2, 1, 3])

    # `context_layer` = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)

    # `context_layer` = [B, F, N, H]
    context_layer = tf.transpose(a=context_layer, perm=[0, 2, 1, 3])

    if do_return_2d_tensor:
        # `context_layer` = [B*F, N*H]
        context_layer = tf.reshape(
            context_layer,
            [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        # `context_layer` = [B, F, N*H]
        context_layer = tf.reshape(
            context_layer,
            [batch_size, from_seq_length, num_attention_heads * size_per_head])

    return context_layer


class Attention(tf.keras.layers.Layer):
    """Attention layer in Transformer"""

    def __init__(self,
                 num_attention_heads,
                 size_per_head,
                 query_act_fn=None,
                 key_act_fn=None,
                 value_act_fn=None,
                 init_range=0.0,
                 dropout=0.0,
                 **kwargs):
        """Initialize Multihead Attention.

        Args:
          num_attention_heads: int, number of attention heads.
          size_per_head: int, hidden size of each attention head.
          query_act_fn: activation function for query.
          key_act_fn: activation function for key.
          value_act_fn: activation function for value.
          init_range: float, range of initializer.
          dropout: float, dropout rate for training.
        """
        super(Attention, self).__init__(**kwargs)
        self._num_attention_heads = num_attention_heads
        self._size_per_head = size_per_head
        self._query_act_fn = query_act_fn
        self._key_act_fn = key_act_fn
        self._value_act_fn = value_act_fn
        self._init_range = init_range
        self._dropout = dropout

        self._hidden_size = self._num_attention_heads * self._size_per_head

    def build(self, input_shape):
        self._query_layer = tf.keras.layers.Dense(
            self._hidden_size,
            activation=get_activation(self._query_act_fn),
            kernel_initializer=create_initializer(self._init_range),
            name="query_layer")
        self._key_layer = tf.keras.layers.Dense(
            self._hidden_size,
            activation=get_activation(self._key_act_fn),
            kernel_initializer=create_initializer(self._init_range),
            name="key_layer")
        self._value_layer = tf.keras.layers.Dense(
            self._hidden_size,
            activation=get_activation(self._query_act_fn),
            kernel_initializer=create_initializer(self._init_range),
            name="value_layer")
        super(Attention, self).build(input_shape)

    def get_config(self):
        return {
            "num_attention_heads": self._num_attention_heads,
            "size_per_head": self._size_per_head,
            "query_act_fn": self._query_act_fn,
            "key_act_fn": self._key_act_fn,
            "value_act_fn": self._value_act_fn,
            "init_range": self._init_range,
            "dropout": self._dropout,
        }

    def call(self, from_tensor, to_tensor, attention_mask=None, training=None):
        """Return outputs of the feedforward network.

        Args:
          x: tensor with shape [batch_size, seq_length, hidden_size]
          training: boolean, whether in training mode or not.

        Returns:
          Output of the feedforward network.
          tensor with shape [batch_size, seq_length, hidden_size]
        """
        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head`

        N = self._num_attention_heads
        H = self._size_per_head

        # Retrieve dynamically known shapes
        from_shape = from_tensor.shape.as_list()
        to_shape = to_tensor.shape.as_list()
        B = from_shape[0]
        F = from_shape[1]
        T = to_shape[1]

        # Input dimensions must be [..., N*H]
        if from_shape[-1] != self._hidden_size or to_shape[-1] != self._hidden_size:
            raise ValueError("Input to the attention layer must have the last dimension "
                             "equals the hidden size")

        # Going through the dense layers first
        query_layer = self._query_layer(from_tensor)
        key_layer = self._key_layer(to_tensor)
        value_layer = self._value_layer(to_tensor)
        # Shape:
        #   query_layer: [B, F, N*H]
        #   key_layer:   [B, T, N*H]
        #   value_layer: [B, T, N*H]

        # Calculate attention score
        query_layer = tf.reshape(query_layer, [B, F, N, H])
        key_layer = tf.reshape(key_layer, [B, T, N, H])
        attention_scores = tf.einsum('BFNH,BTNH->BNFT', query_layer, key_layer)
        attention_scores = tf.multiply(attention_scores,
                                       1.0 / math.sqrt(float(self._size_per_head)))
        # Shape:
        #   attention_scores: [B, N, F, T]

        # Apply attention mask
        if attention_mask is not None:
            # attention_mask: [B, F, T]
            attention_mask = tf.expand_dims(attention_mask, axis=[1])
            # attention_mask: [B, 1, F, T]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_scores += adder

        # Normalize the attention scores to probabilities.
        # `attention_probs` = [B, N, F, T]
        attention_probs = tf.nn.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        if training:
            attention_probs = tf.nn.dropout(
                attention_probs, rate=self._dropout)

        # Calculate context
        value_layer = tf.reshape(value_layer, [B, T, N, H])
        context_layer = tf.einsum(
            'BNFT,BTNH->BFNH', attention_probs, value_layer)
        context_layer = tf.reshape(context_layer, [B, F, N*H])

        return context_layer


class FeedForwardNetwork(tf.keras.layers.Layer):
    """Fully connected feedforward network in Transformer"""

    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 intermediate_act_fn='gelu',
                 init_range=0.0,
                 dropout=0.0,
                 **kwargs):
        """Initialize FeedForwardNetwork.

        Args:
          hidden_size: int, output dim of hidden layer.
          intermediate_size: int, filter size for the inner (first) dense layer.
          intermdeiate_act_fn: activiation function for intermediate layer.
          init_range: float, range of initializer.
          dropout: float, dropout rate for training.
        """
        super(FeedForwardNetwork, self).__init__(**kwargs)
        self._hidden_size = hidden_size
        self._intermediate_size = intermediate_size
        self._intermediate_act_fn = intermediate_act_fn
        self._init_range = init_range
        self._dropout = dropout

    def build(self, input_shape):
        self._filter_dense_layer = tf.keras.layers.Dense(
            self._intermediate_size,
            activation=get_activation(self._intermediate_act_fn),
            kernel_initializer=create_initializer(self._init_range),
            name="filter_layer")
        self._output_dense_layer = tf.keras.layers.Dense(
            self._hidden_size,
            kernel_initializer=create_initializer(self._init_range),
            name="output_layer")
        super(FeedForwardNetwork, self).build(input_shape)

    def get_config(self):
        return {
            "hidden_size": self._hidden_size,
            "intermediate_size": self._intermediate_size,
            "intermediate_act_fn": self._intermediate_act_fn,
            "init_range": self._init_range,
            "dropout": self._dropout,
        }

    def call(self, x, training=None):
        """Return outputs of the feedforward network.

        Args:
          x: tensor with shape [batch_size, seq_length, hidden_size]
          training: boolean, whether in training mode or not.

        Returns:
          Output of the feedforward network.
          tensor with shape [batch_size, seq_length, hidden_size]
        """
        # Retrieve dynamically known shapes
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]

        output = self._filter_dense_layer(x)
        output = self._output_dense_layer(output)
        if training:
            output = tf.nn.dropout(output, rate=self._dropout)

        return output


class TransformerEncoder(tf.keras.layers.Layer):
    """
    Multi-headed transformer encoder layer from "Attention is All You Need".

    This is almost an exact implementation of the original Transformer encoder.

    See the original paper:
    https://arxiv.org/abs/1706.03762

    Also see:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
    """

    """

    Args:
    TODO: Update these below
      input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
      attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
        seq_length], with 1 for positions that can be attended to and 0 in
        positions that should not be.
      hidden_size: int. Hidden size of the Transformer.
      num_hidden_layers: int. Number of layers (blocks) in the Transformer.
      num_attention_heads: int. Number of attention heads in the Transformer.
      intermediate_size: int. The size of the "intermediate" (a.k.a., feed
        forward) layer.
      intermediate_act_fn: function. The non-linear activation function to apply
        to the output of the intermediate/feed-forward layer.
      hidden_dropout_prob: float. Dropout probability for the hidden layers.
      attention_probs_dropout_prob: float. Dropout probability of the attention
        probabilities.
      initializer_range: float. Range of the initializer (stddev of truncated
        normal).
      do_return_all_layers: Whether to also return all layers or just the final
        layer.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size], the final
      hidden layer of the Transformer.

    Raises:
      ValueError: A Tensor shape or parameter is invalid.
    """

    def __init__(self,
                 num_attention_heads=12,
                 hidden_size=768,
                 intermediate_size=1024,
                 intermediate_act_fn='gelu',
                 attn_probs_dropout_prob=0.1,
                 other_dropout_prob=0.1,
                 init_range=0.02,
                 **kwargs):
        """Initialize Transformer Encoder

        Args:
          num_attention_heads: int, number of attention heads
          hidden_size: int, total hidden size. It should be divisable by num_attention_heads
          intermediate_size: int, the size of the intermediate feed forward network
          attn_probs_dropout_prob: float, the dropout probability of the attention probability
          other_dropout_prob: float, the dropout probability for all other places
          init_range: float, range of initializer
        """
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))

        super(TransformerEncoder, self).__init__(**kwargs)
        self._num_attention_heads = num_attention_heads
        self._hidden_size = hidden_size
        self._intermediate_size = intermediate_size
        self._intermediate_act_fn = intermediate_act_fn
        self._attn_probs_dropout_prob = attn_probs_dropout_prob
        self._other_dropout_prob = other_dropout_prob
        self._init_range = init_range

        self._size_per_head = int(hidden_size / num_attention_heads)

    def build(self, input_shape):
        blocks = {}
        blocks['attention'] = Attention(
            num_attention_heads=self._num_attention_heads,
            size_per_head=self._size_per_head,
            query_act_fn='gelu',
            key_act_fn='gelu',
            value_act_fn='gelu',
            init_range=self._init_range,
            dropout=self._attn_probs_dropout_prob,
            name="attn")
        blocks['attention_output'] = tf.keras.layers.Dense(
            self._hidden_size,
            kernel_initializer=create_initializer(self._init_range),
            name="attn_output")
        blocks['attention_norm'] = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-12,
            name="attn_norm")
        blocks['ffn_filter'] = tf.keras.layers.Dense(
            self._intermediate_size,
            activation=get_activation(self._intermediate_act_fn),
            kernel_initializer=create_initializer(self._init_range),
            name="ffn_filter")
        blocks['ffn_output'] = tf.keras.layers.Dense(
            self._hidden_size,
            kernel_initializer=create_initializer(self._init_range),
            name="ffn_output")
        blocks['ffn_norm'] = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-12,
            name="ffn_norm")
        self._blocks = blocks
        super(TransformerEncoder, self).build(input_shape)

    def get_config(self):
        return {
            "num_attention_heads": self._num_attention_heads,
            "hidden_size": self._hidden_size,
            "intermediate_size": self._intermediate_size,
            "intermediate_act_fn": self._intermediate_act_fn,
            "attn_probs_dropout_prob": self._attn_probs_dropout_prob,
            "other_dropout_prob": self._other_dropout_prob,
            "init_range": self._init_range
        }

    def call(self, input_tensor, attention_mask=None, training=None):
        """Return outputs of the feedforward network.

        Args:
          input_tensor: tensor with shape [batch_size, seq_length, hidden_size]
          attention_mask: tensor with shape [batch_size, seq_length, seq_length]
          training: boolean, whether in training mode or not

        Returns:
          Output of the Transformer Encoder Stack
          tensor with shape [batch_size, seq_length, hidden_size]
        """
        # Retrieve dynamically known shapes
        # batch_size = tf.shape(input_tensor)[0]
        # seq_length = tf.shape(input_tensor)[1]

        layer_input = input_tensor
        blocks = self._blocks

        # Attention Layer (self attention)
        attention_output = blocks['attention'](layer_input, layer_input,
                                               attention_mask=attention_mask,
                                               training=training)

        # Attention Output and Dropout
        attention_output = blocks['attention_output'](attention_output)
        if training:
            attention_output = tf.nn.dropout(attention_output,
                                             rate=self._other_dropout_prob)

        # Add and Normalize
        attention_output = blocks['attention_norm'](
            attention_output + layer_input)

        # FeedForward Layer
        ffn_output = blocks['ffn_filter'](attention_output)

        # FeedForward Output and Dropout
        ffn_output = blocks['ffn_output'](ffn_output)
        if training:
            ffn_output = tf.nn.dropout(ffn_output,
                                       rate=self._other_dropout_prob)

        # Add and Normalize
        layer_output = blocks['ffn_norm'](ffn_output + attention_output)

        return layer_output


def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False,
                      training=False):
    """Multi-headed, multi-layer Transformer from "Attention is All You Need".

    This is almost an exact implementation of the original Transformer encoder.

    See the original paper:
    https://arxiv.org/abs/1706.03762

    Also see:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

    Args:
      input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
      attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
        seq_length], with 1 for positions that can be attended to and 0 in
        positions that should not be.
      hidden_size: int. Hidden size of the Transformer.
      num_hidden_layers: int. Number of layers (blocks) in the Transformer.
      num_attention_heads: int. Number of attention heads in the Transformer.
      intermediate_size: int. The size of the "intermediate" (a.k.a., feed
        forward) layer.
      intermediate_act_fn: function. The non-linear activation function to apply
        to the output of the intermediate/feed-forward layer.
      hidden_dropout_prob: float. Dropout probability for the hidden layers.
      attention_probs_dropout_prob: float. Dropout probability of the attention
        probabilities.
      initializer_range: float. Range of the initializer (stddev of truncated
        normal).
      do_return_all_layers: Whether to also return all layers or just the final
        layer.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size], the final
      hidden layer of the Transformer.

    Raises:
      ValueError: A Tensor shape or parameter is invalid.
    """
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (hidden_size, num_attention_heads))

    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]

    # The Transformer performs sum residuals on all layers so the input needs
    # to be the same as the hidden size.
    if input_width != hidden_size:
        raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                         (input_width, hidden_size))

    # We keep the representation as a 2D tensor to avoid re-shaping it back and
    # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
    # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
    # help the optimizer.
    # prev_output = reshape_to_matrix(input_tensor)
    prev_output = input_tensor

    all_layer_outputs = []
    for layer_idx in range(num_hidden_layers):
        with tf.compat.v1.variable_scope("layer_%d" % layer_idx):
            layer_input = prev_output

            attention_output = Attention(num_attention_heads=num_attention_heads,
                                         size_per_head=attention_head_size,
                                         query_act_fn='gelu',
                                         key_act_fn='gelu',
                                         value_act_fn='gelu',
                                         init_range=initializer_range,
                                         dropout=attention_probs_dropout_prob,
                                         name="attn_%d" % layer_idx)(layer_input, layer_input, attention_mask=attention_mask, training=training)

            # Run a linear projection of `hidden_size` then add a residual
            # with `layer_input`.
            # with tf.compat.v1.variable_scope("attention_output"):
            attention_output = tf.keras.layers.Dense(
                hidden_size,
                kernel_initializer=create_initializer(initializer_range),
                name="attention_output")(attention_output)

            # attention_output = tf.compat.v1.layers.dense(
            #     attention_output,
            #     hidden_size,
            #     kernel_initializer=create_initializer(initializer_range))
            attention_output = dropout(
                attention_output, hidden_dropout_prob)
            attention_output = layer_norm(attention_output + layer_input)

            # with tf.compat.v1.variable_scope("attention"):
            # attention_heads = []
            # with tf.compat.v1.variable_scope("self"):
            #     attention_head = attention_layer(
            #         from_tensor=layer_input,
            #         to_tensor=layer_input,
            #         attention_mask=attention_mask,
            #         num_attention_heads=num_attention_heads,
            #         size_per_head=attention_head_size,
            #         attention_probs_dropout_prob=attention_probs_dropout_prob,
            #         initializer_range=initializer_range,
            #         do_return_2d_tensor=True,
            #         batch_size=batch_size,
            #         from_seq_length=seq_length,
            #         to_seq_length=seq_length)
            #     attention_heads.append(attention_head)

            # attention_output = None
            # if len(attention_heads) == 1:
            #     attention_output = attention_heads[0]
            # else:
            #     # In the case where we have other sequences, we just concatenate
            #     # them to the self-attention head before the projection.
            #     attention_output = tf.concat(attention_heads, axis=-1)

            # # Run a linear projection of `hidden_size` then add a residual
            # # with `layer_input`.
            # with tf.compat.v1.variable_scope("output"):
            #     attention_output = tf.compat.v1.layers.dense(
            #         attention_output,
            #         hidden_size,
            #         kernel_initializer=create_initializer(initializer_range))
            #     attention_output = dropout(
            #         attention_output, hidden_dropout_prob)
            #     attention_output = layer_norm(
            #         attention_output + layer_input)

            # # The activation is only applied to the "intermediate" hidden layer.
            # with tf.compat.v1.variable_scope("intermediate"):
            #     intermediate_output = tf.compat.v1.layers.dense(
            #         attention_output,
            #         intermediate_size,
            #         activation=intermediate_act_fn,
            #         kernel_initializer=create_initializer(initializer_range))

            # # Down-project back to `hidden_size` then add the residual.
            # with tf.compat.v1.variable_scope("output"):
            #     layer_output = tf.compat.v1.layers.dense(
            #         intermediate_output,
            #         hidden_size,
            #         kernel_initializer=create_initializer(initializer_range))
            #     layer_output = dropout(layer_output, hidden_dropout_prob)

            layer_output = FeedForwardNetwork(hidden_size=hidden_size,
                                              intermediate_size=intermediate_size,
                                              intermediate_act_fn=intermediate_act_fn,
                                              init_range=initializer_range,
                                              dropout=hidden_dropout_prob,
                                              name="ffn_%d" % layer_idx)(attention_output, training=training)

            # with tf.compat.v1.variable_scope("encoder_output"):
            layer_output = layer_norm(layer_output + attention_output)
            prev_output = layer_output
            all_layer_outputs.append(layer_output)

    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = reshape_from_matrix(layer_output, input_shape)
            final_outputs.append(final_output)
        return final_outputs
    else:
        final_output = reshape_from_matrix(prev_output, input_shape)
        return final_output


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.

    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(input=tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                         (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
    """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
    if len(orig_shape_list) == 2:
        return output_tensor

    output_shape = get_shape_list(output_tensor)

    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.

    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.compat.v1.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))
