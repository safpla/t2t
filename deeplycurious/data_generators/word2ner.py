"""
Problem definition for word to ner definition.
"""

import os
import pickle as pkl

from tensor2tensor.data_generators import problem
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
from deeplycurious.data_generators import generator
from deeplycurious.data_generators.text_encoder import ChineseIndexEncoderPickle
from deeplycurious.data_generators.text_encoder import IOBELabelEncoder

@registry.register_problem()
class Word2ner(problem.Text2TextProblem):
    """Problem spec for Chinese word to named entity recognition definition"""
    @property
    def is_character_level(self):
        return True

    @property
    def targeted_vocab_size(self):
        return 6

    @property
    def vocab_name(self):
        return "vocab.chn"

    @property
    def input_space_id(self):
        return problem.SpaceID.ZH_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.IMAGE_LABEL

    @property
    def num_shards(self):
        return 8

    @property
    def use_subword_tokenizer(self):
        """ (TODO) see what if not implemented"""
        return False

    def generator(self, data_dir, tmp_dir, train):
        vocab_filename = os.path.join(data_dir, self.vocab_name)
        source_vocab = ChineseIndexEncoderPickle(vocab_filename)
        target_vocab = IOBELabelEncoder()
        tag = 'train' if train else 'valid'
        word_file = os.path.join(tmp_dir, 'word_%s' % tag)
        ner_file = os.path.join(tmp_dir, 'ner_%s' % tag)
        EOS = 1
        return generator.chinese_generator(word_file, ner_file, source_vocab, target_vocab, EOS)

    def feature_encoders(self, data_dir):
        vocab_filename = os.path.join(data_dir, self.vocab_name)
        source_encoder = ChineseIndexEncoderPickle(vocab_filename)
        target_encoder = IOBELabelEncoder()
        if self.has_inputs:
            return {"inputs": source_encoder, "targets":target_encoder}
        return {"targets": target_encoder}


@registry.register_hparams
def word2ner_hparams():
    hparams = transformer.transformer_base_single_gpu()
    hparams.learning_rate_warmup_steps = 6000
    hparams.num_hidden_layers = 4
    hparams.learning_rate = 0.1
    hparams.hidden_size = 256
    return hparams

@registry.register_hparams
def word2ner_hparams_long():
    hparams = transformer.transformer_base_single_gpu()
    hparams.learning_rate_warmup_steps = 16000
    hparams.num_hidden_layers = 4
    hparams.learning_rate = 0.03
    hparams.hidden_size = 128
    hparams.filter_size = 1024
    hparams.max_length = 512
    return hparams
