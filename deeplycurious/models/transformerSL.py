# coding=utf-8
# copyright 2017 Xu Haowen
#
# email: haowen.will.xu@gmail.com

"""
sequence labeling with attention
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

from tensor2tensor.utils import t2t_model
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import beam_search
from tensor2tensor.utils import registry
from tensor2tensor.utils import expert_utils

import sys, os
import tensorflow as tf


@registry.register_model
class TransformerSL(t2t_model.T2TModel):
    def encode(self, inputs, hparams):
        """Encode TransformerSL inputs.

        Args:
            input: TransformerSL inputs [batch_size, input_length, 1,  hidden_dim]
            hparams: hyperparemters for model.

        Returns:
            Tuple of:
                encoder_output: Encoder representation.
                    [batch_size, input_length, hidden_dim]
                top_layer_attention_bias: Bias and mask weights for
                    top layer. [batch_size, input_length]
        """
        inputs = common_layers.flatten4d3d(inputs)

        encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
            transformer_prepare_encoder(inputs, hparams))

        encoder_input = tf.nn.dropout(
            encoder_input, 1.0 - hparams.layer_prepostprocess_dropout)

        encoder_output = transformer_encoder(
            encoder_input,
            self_attention_bias,
            hparams)
        return encoder_output, encoder_decoder_attention_bias

    def decode(self,
               decoder_input,
               encoder_output,
               encoder_decoder_attention_bias,
               hparams,
               cache=None):
        output = tf.concat([encoder_output, decoder_input], axis=2)
        # TODO(haowen): Add layer_preprocess with given hidden dim
        with tf.variable_scope("ffn"):
            output = transformer_ffn_layer(
                output,
                hparams)
        #return tf.expand_dims(encoder_output, axis=2)
        return tf.expand_dims(output, axis=2)

    def model_fn_body(self, features):
        """TransformerSL main model_fn.

        Args:
            features: Map of features to the model.
            Should contain the following:
                "inputs": TransformerSL inputs
                          [batch_size, input_length, hidden_dim]
                "targets": label sequence
                          [batch_size, input_length, num_class]

        Returns:
            prediction logits. [batch_size, input_length, 1, num_class]
        """
        hparams = self._hparams

        inputs = features["inputs"]
        encoder_output, encoder_decoder_attention_bias = self.encode(
            inputs, hparams)

        targets = features["targets"]
        targets = common_layers.flatten4d3d(targets)
        targets = common_layers.shift_right_3d(targets)

        return self.decode(targets, encoder_output,
                           encoder_decoder_attention_bias,
                           hparams)

    def infer(self,
              features=None,
              decode_length=0,
              beam_size=1,
              top_beams=1,
              last_position_only=False,
              alpha=0.0):
        """A inference method.

        Args:
            features: an map of string to 'Tensor'
            decode_length: an interger.
            beam_size: number of beams.
            top_beams: an integer. How many of the beams to return.
            last_position_only: a boolean, speed-up by computing last position only.
            alpha: Folat that controls the length penalty. larger the alpha,
                stronger the preference for slonger translations.

        Returns:
            samples: an integer 'Tensor'. [batch_size, max_length]
        """
        if beam_size == 1:
            tf.logging.info("Greedy Decoding")
            samples, _, _ = self._greedy_infer(features,
                                              decode_length,
                                              last_position_only)
        else:
            tf.logging.info("Beam Decoding with beam size %d" % beam_size)
            raise NotImplementedError("Abstract Method")

        return samples

    def _greedy_infer(self,
                      features,
                      decode_length,
                      last_position_only=True,
                      beam_size=1,
                      top_beams=1,
                      alpha=1.0):
        """greedy decoding.

        Args:
            features: an map of string to 'Tensor'
            decode_length: an integer. How many addtional timesteps to decode.
            last_position_only: a boolean, speed-up by computing last position only.

        Returns:
            samples: [batch_size, input_length]
            logits: Not returned
            losses: Not returned

        Raises:
            NotImplementedError: If there are multiple data shards.
        """
        if self._num_datashards != 1:
            raise NotImplementedError("Fast decoding only supports a single shard.")
        dp = self._data_parallelism
        hparams = self._hparams
        inputs = features["inputs"]
        batch_size = tf.shape(inputs)[0]
        target_modality = self._problem_hparams.target_modality
        decode_length = tf.shape(inputs)[1]
        inputs = tf.expand_dims(inputs, axis=1)
        if len(inputs.shape) < 5:
            inputs = tf.expand_dims(inputs, axis=4)
        s = tf.shape(inputs)
        inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])

        inputs = self._shard_features({"inputs": inputs})["inputs"]
        input_modality = self._problem_hparams.input_modality["inputs"]
        with tf.variable_scope(input_modality.name):
            inputs = input_modality.bottom_sharded(inputs, dp)
        with tf.variable_scope("body"):
            encoder_output, encoder_decoder_attention_bias = dp(
                self.encode, inputs, hparams)
        encoder_output = encoder_output[0]
        encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]
        if hparams.pos == "timing":
            timing_signal = common_attention.get_timing_signal_1d(
                decode_length + 1, hparams.hidden_size)

        def preprocess_targets(targets, i):
            """Performs preprocessing steps on the targets to prepare for the decoder.

            This includes:
              - Embedding the ids.
              - Flattening to 3D tensor.
              - Optionally adding timing signals.

            Args:
              targets: inputs ids to the decoder. [batch_size, 1]
              i: scalar, Step number of the decoding loop.

            Returns:
              Processed targets [batch_size, 1, hidden_dim]
            """
            # _shard_features called to ensure that the variable names match
            targets = self._shard_features({"targets": targets})["targets"]
            with tf.variable_scope(target_modality.name):
                targets = target_modality.targets_bottom_sharded(targets, dp)[0]
            targets = common_layers.flatten4d3d(targets)

            # TODO(llion): Explain! Is this even needed?
            targets = tf.cond(
                tf.equal(i, 0),
                lambda: tf.zeros_like(targets),
                lambda: targets)

            if hparams.pos == "timing":
                targets += timing_signal[:, i:i+1]
            return targets

        def symbols_to_logits_fn(ids, i):
            """
            From ids to logits for next symbol.
            """
            ids = ids[:, -1:]
            targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
            targets = preprocess_targets(targets, i)
            with tf.variable_scope("body"):
                body_outputs = dp(
                    self.decode,
                    targets,
                    encoder_output[:,i:i+1,:],
                    encoder_decoder_attention_bias[:,:,:,i:i+1],
                    hparams)
            with tf.variable_scope(target_modality.name):
                logits = target_modality.top_sharded(body_outputs, None, dp)[0]
            return tf.squeeze(logits, axis=[1, 2, 3])

        if beam_size > 1: # Beam Search
            raise NotImplementedError("Beam search not implemented")
        else: # Greedy
            def inner_loop(i, next_id, decoded_ids):
                logits = symbols_to_logits_fn(next_id, i)
                next_id = tf.expand_dims(tf.argmax(logits, axis=-1), axis=1)
                decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
                return i+1, next_id, decoded_ids

            decoded_ids = tf.zeros([batch_size, 0], dtype=tf.int64)
            next_id = tf.zeros([batch_size, 1], dtype=tf.int64)
            _, _, decoded_ids = tf.while_loop(
                lambda i, *_: tf.less(i, decode_length),
                inner_loop,
                [tf.constant(0), next_id, decoded_ids],
                shape_invariants=[
                    tf.TensorShape([]),
                    tf.TensorShape([None, None]),
                    tf.TensorShape([None, None]),
                ])
        return decoded_ids, None, None


def transformer_encoder(encoder_input,
                        encoder_self_attention_bias,
                        hparams,
                        name="encoder"):
    """A stack of transormer layer.

    Args:
        encoder_input: a Tensor [batch_size, input_length, hidden_dim]
        encoder_self_attention_bias: bias Tensor for sel-attention
            (see common_attention.attention_bias())
        hparams: hyperparameters
        name: a string

    Returns:
        y: a Tensor [batch_size, input_length, hidden_dim]
    """
    x = encoder_input
    with tf.variable_scope(name):
        pad_remover = None
        if hparams.use_pad_remover:
            pad_remover = expert_utils.PadRemover(
                common_attention.attention_bias_to_padding(
                    encoder_self_attention_bias))
        for layer in xrange(hparams.num_encoder_layers or
                            hparams.num_hidden_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    y = common_attention.multihead_attention(
                        common_layers.layer_preprocess(x, hparams),
                        None,
                        encoder_self_attention_bias,
                        hparams.attention_key_channels or hparams.hidden_size,
                        hparams.attention_value_channels or hparams.hidden_size,
                        hparams.hidden_size,
                        hparams.num_heads,
                        hparams.attention_dropout,
                        attention_type=hparams.self_attention_type,
                        max_relative_position=hparams.max_relative_position)
                    x = common_layers.layer_postprocess(x, y, hparams)
                with tf.variable_scope("ffn"):
                    y = transformer_ffn_layer(
                        common_layers.layer_preprocess(x, hparams),
                        hparams,
                        pad_remover)
                    x = common_layers.layer_postprocess(x, y, hparams)

        return common_layers.layer_preprocess(x, hparams)

def transformer_prepare_encoder(inputs, hparams):
    """Prepare one shard of the model for the encoder.

    Args:
        inputs: [batch_size, input_length, hidden_dim]
        hparams: hyperparameters

    Returns:
        encoder_input: a Tensor, bottom of encoder stack
            [batch_size, input_length, hidden_dim]
        encoder_self_attention_bias: a bias tensor for use in encoder
            self-attention [batch_size, input_length]
        top_layer_attention_bias: a bias tensor for use in top layer
            classification [batch_size, input_length]
    """
    ishape_static = inputs.shape.as_list()
    encoder_input = inputs
    encoder_padding = common_attention.embedding_to_padding(encoder_input)
    ignore_padding = common_attention.attention_bias_ignore_padding(
        encoder_padding)
    encoder_self_attention_bias = ignore_padding
    top_layer_attention_bias = ignore_padding
    if hparams.proximity_bias:
        encoder_self_attention_bias += common_attention.attention_bias_proximal(
            tf.shape(inputs)[1])
    if hparams.pos == "timing":
        encoder_input = common_attention.add_timing_signal_1d(encoder_input)
    return (encoder_input, encoder_self_attention_bias,
            top_layer_attention_bias)

def transformer_ffn_layer(x, hparams, pad_remover=None):
    """Feed-forward layer in the transformer.

    Args:
      x: a Tensor of shape [batch_size, length, hparams.hidden_size]
      hparams: hyperparmeters for model
      pad_remover: an expert_utils.PadRemover object tracking the padding
        positions. If provided, when using convolutional settings, the padding
        is removed before applying the convolution, and restored afterward. This
        can give a significant speedup.

    Returns:
      a Tensor of shape [batch_size, length, hparams.hidden_size]
    """
    if hparams.ffn_layer == "conv_hidden_relu":
      # In simple convolution mode, use `pad_remover` to speed up processing.
      if pad_remover:
        original_shape = tf.shape(x)
        # Collapse `x` across examples, and remove padding positions.
        x = tf.reshape(x, tf.concat([[-1], tf.shape(x)[2:]], axis=0))
        x = tf.expand_dims(pad_remover.remove(x), axis=0)
      conv_output = common_layers.conv_hidden_relu(
          x,
          hparams.filter_size,
          hparams.hidden_size,
          dropout=hparams.relu_dropout)
      if pad_remover:
        # Restore `conv_output` to the original shape of `x`, including padding.
        conv_output = tf.reshape(
            pad_remover.restore(tf.squeeze(conv_output, axis=0)), original_shape)
      return conv_output
    elif hparams.ffn_layer == "parameter_attention":
      return common_attention.parameter_attention(
          x, hparams.parameter_attention_key_channels or hparams.hidden_size,
          hparams.parameter_attention_value_channels or hparams.hidden_size,
          hparams.hidden_size, hparams.filter_size, hparams.num_heads,
          hparams.attention_dropout)
    elif hparams.ffn_layer == "conv_hidden_relu_with_sepconv":
      return common_layers.conv_hidden_relu(
          x,
          hparams.filter_size,
          hparams.hidden_size,
          kernel_size=(3, 1),
          second_kernel_size=(31, 1),
          padding="LEFT",
          dropout=hparams.relu_dropout)
    else:
      assert hparams.ffn_layer == "none"
      return x

