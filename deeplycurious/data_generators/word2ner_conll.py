"""
Problem definition for word to ner definitioni.
"""

import os, sys
import pickle as pkl

father_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(father_dir)
from config.path_config_conll import *

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
from deeplycurious.data_generators import generator
from deeplycurious.data_generators.text_encoder import EnglishIndexEncoderPickleConll
from deeplycurious.data_generators.text_encoder import LabelEncoderConll

@registry.register_problem()
class Word2ner_conll(problem.Text2TextProblem):
    """Problem spec for Chinese word to named entity recognition definition"""
    @property
    def is_character_level(self):
        return False

    @property
    def targeted_vocab_size(self):
        return 10

    @property
    def vocab_name(self):
        return "vocab_conll.chn"

    @property
    def input_space_id(self):
        return problem.SpaceID.EN_TOK

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
        source_vocab = EnglishIndexEncoderPickleConll(vocab_filename)
        target_vocab = LabelEncoderConll()
        tag = 'train' if train else 'valid'
        word_file = os.path.join(tmp_dir, 'word_%s' % tag)
        ner_file = os.path.join(tmp_dir, 'ner_%s' % tag)
        EOS = 1
        obj = self.hparams
        return generator.chinese_generator(word_file, ner_file, source_vocab,
                                           target_vocab, EOS,
                                           one_hot_feature=USE_ONE_HOT_FEATURE)

    def generate_data(self, data_dir, tmp_dir, task_id=-1):
        """ TODO """
        train_paths = self.training_filepaths(
            data_dir, self.num_shards, shuffled=False)
        dev_paths = self.dev_filepaths(
            data_dir, self.num_dev_shards, shuffled=False)
        if self.use_train_shards_for_dev:
            all_paths = train_paths + dev_paths
            generator_utils.generate_files(
                self.generator(data_dir, tmp_dir, True), all_paths)
            generator_utils.shuffle_dataset(all_paths)
        else:
            generator_utils.generate_dataset_and_shuffle(
                self.generator(data_dir, tmp_dir, True), train_paths,
                self.generator(data_dir, tmp_dir, False), dev_paths)

    def feature_encoders(self, data_dir):
        vocab_filename = os.path.join(data_dir, self.vocab_name)
        source_encoder = EnglishIndexEncoderPickleConll(vocab_filename)
        target_encoder = LabelEncoderConll()
        if self.has_inputs:
            return {"inputs": source_encoder, "targets":target_encoder}
        return {"targets": target_encoder}

    def hparams(self, defaults, unused_model_hparams):
        """ TODO """
        p = defaults
        p.stop_at_eos = int(True)
        if self.has_inputs:
            source_vocab_size = self._encoders["inputs"].vocab_size
            if USE_ONE_HOT_FEATURE:
                p.input_modality = {
                    "inputs": (registry.Modalities.SYMBOL+':extra_onehot_feature',
                               source_vocab_size),
                }
            else:
                p.input_modality = {
                    "inputs": (registry.Modalities.SYMBOL, source_vocab_size)
                }
        target_vocab_size = self._encoders["targets"].vocab_size
        p.target_modality = (registry.Modalities.SYMBOL, target_vocab_size)
        if self.has_inputs:
            p.input_space_   = self.input_space_id
        p.target_space_id = self.target_space_id
        if self.is_character_level:
            p.loss_multiplier = 2.0


@registry.register_hparams
def word2ner_conll_hparams():
    hparams = transformer.transformer_base_single_gpu()
    hparams.learning_rate_warmup_steps = 4000
    hparams.num_hidden_layers = 4
    hparams.learning_rate = 0.1
    hparams.hidden_size = 256
    return hparams

@registry.register_hparams
def word2ner_conll_hparams_small():
    hparams = word2ner_conll_hparams()
    hparams.learning_rate_decay_scheme
    hparams.hidden_size = 128
    hparams.filter_size = 512
    return hparams
@registry.register_hparams
def word2ner_conll_hparams_singlehead():
    hparams = word2ner_conll_hparams()
    hparams.learning_rate_warmup_steps = 4000
    hparams.num_hidden_layers = 4
    hparams.learning_rate = 0.1
    hparams.hidden_size = 128
    hparams.num_heads = 1
    return hparams

