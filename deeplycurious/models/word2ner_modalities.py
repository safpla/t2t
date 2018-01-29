# coding=utf-8
"""
Define modalities used in word2ner problem serial here.
"""

import os, sys
father_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(father_dir)

from tensor2tensor.utils import modality
from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry
from tensor2tensor.utils import expert_utils as eu

import tensorflow as tf
import pickle as pkl
import numpy as np
from config.path_config import pretrained_emb, vocab_file
from config.path_config_conll import ONE_HOT_NUM
@registry.register_symbol_modality("extra_onehot_feature")
class SymbolModalityExtraOnehotFeature(modalities.SymbolModality):
    """Modality for sets of discrete symols, plus extra onehot features.

    Input:
        Embedding + extra onehot features

    Output:
        Linear transformation + softmax.
    """
    @property
    def name(self):
        return "symbol_modality_with_extra_onehot_feature_%d_%d" % (
            self._vocab_size, self._body_input_depth)

    @property
    def _body_input_depth(self):
        return self._model_hparams.hidden_size - ONE_HOT_NUM

    def top_dimensionality(self):
        return self._vocab_size

    def _get_weights(self, use_pretrain=False, dim=None):
        """
        create embedding or use pretrained embedding or create softmax variable

        Returns:
            a list of self._num_shards Tensors.
        """
        if dim is None:
            dim = self._body_input_depth

        num_shards = self._model_hparams.symbol_modality_num_shards
        shards = []
        for i in range(num_shards):
            shard_size = (self._vocab_size // num_shards) + (
                1 if i < self._vocab_size % num_shards else 0)
            var_name = "weights_%d" % i
            shards.append(
                tf.get_variable(
                    var_name, [shard_size, dim],
                    initializer=tf.random_normal_initializer(
                        0.0, dim**-0.5)))
        if num_shards == 1:
            ret = shards[0]
        else:
            ret = tf.concat(shards, 0)
        ret = eu.convert_gradient_to_tensor(ret)
        return ret

    def bottom_simple(self, x, name, reuse):
        with tf.variable_scope(name, reuse=reuse):
            # get index and case_feature
            shape = tf.shape(x)
            x = tf.reshape(x, [shape[0], 2, -1, 1])
            x = tf.transpose(x, [0, 2, 1, 3])
            ind = x[:,:,0,:]
            feature = x[:,:,1,:]
            # Squeeze out the channels dimension.

            var = self._get_weights(use_pretrain=True)
            emb = tf.gather(var, ind)
            one_hot = tf.cast(tf.one_hot(feature, ONE_HOT_NUM), tf.float32)
            ret = tf.concat([emb, one_hot], 3)
            return ret

    def bottom(self, x):
        self._bottom_was_called = True
        if self._model_hparams.shared_embedding_and_softmax_weights:
            return self.bottom_simple(x, "shared", reuse=None)
        else:
            return self.bottom_simple(x, "input_emb", reuse=None)

    def top(self, body_output, _):
        """Generate logits.

        Args:
            body_output: A Tensor with shape [batch, p0, p1, body_input_depth]
        Returns:
            logits: A Tensor with shape [batch, p0, p1, ?, vocab_size].
        """
        if self._model_hparams.shared_embedding_and_softmax_weights:
            scope_name = "shared"
            reuse = True
        else:
            scope_name = "softmax"
            reuse = False
        with tf.variable_scope(scope_name, reuse=reuse):
            var = self._get_weights(dim=self._model_hparams.hidden_size)
            if (self._model_hparams.factored_logits and
                self._model_hparams.mode == tf.estimator.ModeKeys.TRAIN):
                body_output = tf.expand_dims(body_output, 3)
                logits = common_layers.FactoredTensor(body_output, var)
            else:
                shape = tf.shape(body_output)[:-1]
                body_output = tf.reshape(body_output, [-1, self._model_hparams.hidden_size])
                logits = tf.matmul(body_output, var, transpose_b=True)
                logits = tf.reshape(
                    logits, tf.concat([shape, [1, self._vocab_size]], 0))
            return logits


@registry.register_symbol_modality("pretrained_emb")
class SymbolModalityPretrainedEmb(modalities.SymbolModality):
    """Modality for sets of discrete symols, plus extra onehot features.

    Input:
        Embedding

    Output:
        Linear transformation + softmax.
    """
    @property
    def name(self):
        return "symbol_modality_with_pretrained_emb_%d_%d" % (
            self._vocab_size, self._body_input_depth)

    def _get_weights(self, dim=None):
        """
        create embedding or use pretrained embedding or create softmax variable

        Returns:
            a list of self._num_shards Tensors.
        """
        if dim is None:
            dim = self._body_input_depth
        var_name = "weights_"
        emb_file = open(pretrained_emb, 'rb')
        emb = pkl.load(emb_file, encoding='latin1')
        emb_file.close()
        emb = np.asarray(emb, dtype=np.float32)

        num_shards = self._model_hparams.symbol_modality_num_shards
        shards = []
        start_ind = 0
        for i in range(num_shards):
            shard_size = (self._vocab_size // num_shards) + (
                1 if i < self._vocab_size % num_shards else 0)
            var_name = "weights_%d" % i
            init = emb[start_ind:start_ind + shard_size, :]
            start_ind += shard_size
            shards.append(
                tf.get_variable(var_name, initializer=init))
        if num_shards == 1:
            ret = shards[0]
        else:
            ret = tf.concat(shards, 0)
        ret = eu.convert_gradient_to_tensor(ret)
        return ret
