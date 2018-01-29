"""
Problem definition for word to ner definitioni.
"""

import os, sys
import pickle as pkl

from tensor2tensor.data_generators import problem
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
from deeplycurious.data_generators import generator
from deeplycurious.data_generators.text_encoder import ChineseIndexEncoderPickleSubword
from deeplycurious.data_generators.text_encoder import IOBELabelEncoderSubword

father_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(father_dir)
from config.path_config import vocab_file
@registry.register_problem()
class Word2ner_subword(problem.Text2TextProblem):
    """Problem spec for Chinese word to named entity recognition definition"""
    @property
    def is_character_level(self):
        return False

    @property
    def targeted_vocab_size(self):
        return 5

    @property
    def vocab_name(self):
        return "vocab_subword.chn"

    @property
    def input_space_id(self):
        return problem.SpaceID.ZH_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.ZH_TOK

    @property
    def num_shards(self):
        return 1

    @property
    def use_subword_tokenizer(self):
        """ (TODO) see what if not implemented"""
        return False

    def generator(self, data_dir, tmp_dir, train):
        source_vocab = ChineseIndexEncoderPickleSubword(vocab_file)
        target_vocab = IOBELabelEncoderSubword()
        tag = 'train' if train else 'valid'
        word_file = os.path.join(tmp_dir, 'word_%s' % tag)
        ner_file = os.path.join(tmp_dir, 'iobe_%s' % tag)
        EOS = 1
        return generator.chinese_generator(word_file, ner_file, source_vocab, target_vocab, EOS)

    def feature_encoders(self, data_dir):
        source_encoder = ChineseIndexEncoderPickleSubword(vocab_file)
        target_encoder = IOBELabelEncoderSubword()
        if self.has_inputs:
            return {"inputs": source_encoder, "targets":target_encoder}
        return {"targets": target_encoder}

    def hparams(self, defaults, unused_model_hparams):
        """ TODO """
        p = defaults
        p.stop_at_eos = int(True)
        if self.has_inputs:
            source_vocab_size = self._encoders["inputs"].vocab_size
            print('source_vocab_size: ', source_vocab_size)
            #p.input_modality = {
            #    "inputs": (registry.Modalities.SYMBOL+':pretrained_emb',
            #               source_vocab_size),
            #}
            p.input_modality = {
                "inputs": (registry.Modalities.SYMBOL, source_vocab_size)
            }
        target_vocab_size = self._encoders["targets"].vocab_size
        print('target_vocab_size: ', target_vocab_size)
        p.target_modality = (registry.Modalities.SYMBOL, target_vocab_size)
        if self.has_inputs:
            p.input_space_   = self.input_space_id
        p.target_space_id = self.target_space_id
        if self.is_character_level:
            p.loss_multiplier = 2.0


@registry.register_hparams
def word2ner_subword_hparams():
    hparams = transformer.transformer_base_single_gpu()
    hparams.learning_rate_warmup_steps = 6000
    hparams.num_hidden_layers = 4
    hparams.learning_rate = 0.1
    hparams.hidden_size = 128
    #hparams.optimizer = 'Adadelta'
    return hparams

@registry.register_hparams
def word2ner_subword_hparams_long():
    hparams = word2ner_subword_hparams()
    hparams.num_hidden_layers = 4
    hparams.hidden_size = 128
    hparams.max_length = 512
    return hparams

@registry.register_hparams
def word2ner_subword_hparams_small():
    hparams = word2ner_subword_hparams()
    hparams.num_hidden_layers = 4
    hparams.hidden_size = 128
    hparams.max_length = 512
    hparams.filter_size = 1024
    return hparams

@registry.register_hparams
def word2ner_subword_hparams_deep():
    hparams = word2ner_subword_hparams()
    hparams.learning_rate_warmup_steps = 6000
    hparams.num_hidden_layers = 6
    hparams.learning_rate = 0.02
    hparams.hidden_size = 128
    hparams.max_length = 512
    return hparams

@registry.register_hparams
def word2ner_subword_hparams_thin():
    hparams = word2ner_subword_hparams()
    hparams.num_hidden_layers = 4
    hparams.hidden_size = 128
    hparams.max_length = 512
    hparams.num_heads = 4
    return hparams

@registry.register_hparams
def word2ner_subword_hparams_singlehead():
    hparams = word2ner_subword_hparams()
    hparams.num_hidden_layers = 4
    hparams.hidden_size = 64
    hparams.filter_size = 256
    hparams.max_length = 512
    hparams.num_heads = 1
    return hparams

@registry.register_hparams
def word2ner_subword_hparams_deep8():
    hparams = word2ner_subword_hparams()
    hparams.num_hidden_layers = 8
    hparams.hidden_size = 256
    hparams.max_length = 512
    hparams.learning_rate = 0.02
    #hparams.proximity_bias = int(True)
    return hparams

@registry.register_hparams
def word2ner_subword_hparams_deep10():
    hparams = word2ner_subword_hparams()
    hparams.num_hidden_layers = 10
    hparams.hidden_size = 256
    hparams.max_length = 512
    hparams.learning_rate = 0.02
    return hparams

