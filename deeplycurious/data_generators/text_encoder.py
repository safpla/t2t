# coding:utf-8

import pickle as pkl
from tensor2tensor.data_generators.text_encoder import TextEncoder
from tensor2tensor.data_generators.text_encoder import *

class ChineseIndexEncoderPickle(TextEncoder):
    def __init__(self,
                 vocab_filename,
                 reverse=False,
                 vocab_list=None,
                 replace_oov=None,
                 num_reserved_ids=2):
        super(ChineseIndexEncoderPickle, self).__init__(num_reserved_ids=num_reserved_ids)
        self._reverse = reverse
        self._replace_oov = replace_oov
        vocab_stream = open(vocab_filename, 'rb')
        vocab_data = pkl.load(vocab_stream)
        vocab_stream.close()
        self._token_to_id = vocab_data[0]
        self._token_to_id[EOS] = EOS_ID
        self._token_to_id[PAD] = PAD_ID
        self._id_to_token = dict((k,v) for v, k in self._token_to_id.items())
        self._freq = vocab_data[1]

    def encode(self, sentence):
        tokens = sentence
        ids = [self._token_to_id['unknown'] if tok not in self._token_to_id else self._token_to_id[tok] for tok in tokens]
        return ids[::-1] if self._reverse else ids

    def decode(self, ids):
        return ''.join(self.decode_list(ids))

    def decode_list(self, ids):
        seq = reversed(ids) if self._reverse else ids
        return [self._safe_id_to_token(i) for i in ids]

    @property
    def vocab_size(self):
        return len(self._id_to_token)

    def _safe_id_to_token(self, idx):
        return self._id_to_token.get(idx, "ID_%d" %idx)

class ChineseIndexEncoderPickleSubword(TextEncoder):
    def __init__(self,
                 vocab_filename,
                 reverse=False,
                 vocab_list=None,
                 replace_oov=None,
                 num_reserved_ids=2):
        super(ChineseIndexEncoderPickleSubword, self).__init__(num_reserved_ids=num_reserved_ids)
        self._reverse = reverse
        self._replace_oov = replace_oov
        vocab_stream = open(vocab_filename, 'rb')
        vocab_data = pkl.load(vocab_stream)
        vocab_stream.close()
        self._token_to_id = vocab_data[0]
        self._token_to_id[EOS] = EOS_ID
        self._token_to_id[PAD] = PAD_ID
        self._id_to_token = dict((k,v) for v, k in self._token_to_id.items())
        self._freq = vocab_data[1]

    def encode(self, sentence):
        tokens = sentence.strip().split()
        ids = [self._token_to_id['unknown'] if tok not in self._token_to_id else self._token_to_id[tok] for tok in tokens]
        return ids[::-1] if self._reverse else ids

    def decode(self, ids):
        return ' '.join(self.decode_list(ids))

    def decode_list(self, ids):
        seq = reversed(ids) if self._reverse else ids
        return [self._safe_id_to_token(i) for i in ids]

    @property
    def vocab_size(self):
        return len(self._id_to_token)

    def _safe_id_to_token(self, idx):
        return self._id_to_token.get(idx, "ID_%d" %idx)

class EnglishIndexEncoderPickleConll(TextEncoder):
    def __init__(self,
                 vocab_filename,
                 reverse=False,
                 vocab_list=None,
                 replace_oov=None,
                 num_reserved_ids=2):
        super(EnglishIndexEncoderPickleConll, self).__init__(num_reserved_ids=num_reserved_ids)
        self._reverse = reverse
        self._replace_oov = replace_oov
        vocab_stream = open(vocab_filename, 'rb')
        vocab_data = pkl.load(vocab_stream)
        vocab_stream.close()
        self._token_to_id = vocab_data[0]
        self._id_to_token = dict((k,v) for v, k in self._token_to_id.items())
        self._freq = vocab_data[1]

    def encode(self, sentence):
        tokens = sentence.strip().split()
        ids = [self._token_to_id['<unk>'] if tok not in self._token_to_id else self._token_to_id[tok] for tok in tokens]
        return ids[::-1] if self._reverse else ids

    def decode(self, ids):
        return ' '.join(self.decode_list(ids))

    def decode_list(self, ids):
        seq = reversed(ids) if self._reverse else ids
        return [self._safe_id_to_token(i) for i in ids]

    @property
    def vocab_size(self):
        return len(self._id_to_token)

    def _safe_id_to_token(self, idx):
        return self._id_to_token.get(idx, "ID_%d" %idx)

class IOBELabelEncoder(TextEncoder):
    def __init__(self,
                 reverse=False,
                 vocab_list=None,
                 replace_oov=None,
                 num_reserved_ids=2):
        super(IOBELabelEncoder, self).__init__(num_reserved_ids=num_reserved_ids)
        self._reverse = reverse
        self._replace_oov = replace_oov
        self._token_to_id = {'I':2, 'O':3, 'B':4, 'E':5}
        self._id_to_token = {2:'I', 3:'O', 4:'B', 5:'E'}

    def encode(self, sentence):
        ids = [self._token_to_id[tok] for tok in sentence]
        return ids[::-1] if self._reverse else ids

    def decode(self, ids):
        return ''.join(self.decode_list(ids))

    def decode_list(self, ids):
        seq = reversed(ids) if self._reverse else ids
        return [self._safe_id_to_token(i) for i in ids]

    def _safe_id_to_token(self, idx):
        return self._id_to_token.get(idx, "ID_%d" % idx)

    @property
    def vocab_size(self):
        return len(self._id_to_token) + 2

class IOBELabelEncoderSubword(TextEncoder):
    def __init__(self,
                 reverse=False,
                 vocab_list=None,
                 replace_oov=None,
                 num_reserved_ids=2):
        super(IOBELabelEncoderSubword, self).__init__(num_reserved_ids=num_reserved_ids)
        self._reverse = reverse
        self._replace_oov = replace_oov
        self._token_to_id = {'I':2, 'O':3, 'B':4, 'E':5}
        self._token_to_id[EOS] = EOS_ID
        self._token_to_id[PAD] = PAD_ID
        self._id_to_token = dict((k, v) for v, k in self._token_to_id.items())

    def encode(self, sentence):
        sentence = sentence.strip().split()
        ids = [self._token_to_id[tok] for tok in sentence]
        return ids[::-1] if self._reverse else ids

    def decode(self, ids):
        return ' '.join(self.decode_list(ids))

    def decode_list(self, ids):
        seq = reversed(ids) if self._reverse else ids
        return [self._safe_id_to_token(i) for i in ids]

    def _safe_id_to_token(self, idx):
        return self._id_to_token.get(idx, "ID_%d" % idx)

    @property
    def vocab_size(self):
        return len(self._id_to_token)

class LabelEncoderConll(TextEncoder):
    def __init__(self,
                 reverse=False,
                 vocab_list=None,
                 replace_oov=None,
                 num_reserved_ids=2):
        super(LabelEncoderConll, self).__init__(num_reserved_ids=num_reserved_ids)
        self._reverse = reverse
        self._replace_oov = replace_oov
        self._token_to_id ={'B-LOC': 9, 'I-MISC': 5, 'O': 2, 'B-MISC': 8,
                            'I-LOC': 3, 'I-ORG': 6, 'B-ORG': 7, 'I-PER': 4}
        self._id_to_token = {2:'O', 3:'I-LOC', 4:'I-PER', 5:'I-MISC', 6:'I-ORG',
                             7:'B-ORG', 8:'B-MISC', 9:'B-LOC'}

    def encode(self, sentence):
        sentence = sentence.strip().split()
        ids = [self._token_to_id[tok] for tok in sentence]
        return ids[::-1] if self._reverse else ids

    def decode(self, ids):
        return ' '.join(self.decode_list(ids))

    def decode_list(self, ids):
        seq = reversed(ids) if self._reverse else ids
        return [self._safe_id_to_token(i) for i in ids]

    def _safe_id_to_token(self, idx):
        return self._id_to_token.get(idx, "ID_%d" % idx)

    @property
    def vocab_size(self):
        return len(self._id_to_token) + 2

class LabelEncoderMSRA(TextEncoder):
    def __init__(self,
                 reverse=False,
                 vocab_list=None,
                 replace_oov=None,
                 num_reserved_ids=2):
        super(LabelEncoderMSRA, self).__init__(num_reserved_ids=num_reserved_ids)
        self._reverse = reverse
        self._replace_oov = replace_oov
        self._token_to_id ={'I-PER':2, 'B-PER':3, 'E-PER':4,
                            'I-ORG':5, 'B-ORG':6, 'E-ORG':7,
                            'I-LOC':8, 'B-LOC':9, 'E-LOC':10, 'O':11}
        self._id_to_token = {v:k for k,v in self._token_to_id.items()}

    def encode(self, sentence):
        sentence = sentence.strip().split()
        ids = [self._token_to_id[tok] for tok in sentence]
        return ids[::-1] if self._reverse else ids

    def decode(self, ids):
        return ' '.join(self.decode_list(ids))

    def decode_list(self, ids):
        seq = reversed(ids) if self._reverse else ids
        return [self._safe_id_to_token(i) for i in ids]

    def _safe_id_to_token(self, idx):
        return self._id_to_token.get(idx, "ID_%d" % idx)

    @property
    def vocab_size(self):
        return len(self._id_to_token) + 2


if __name__ == '__main__':
    #encoder = ChineseIndexEncoderPickle('/home/xuhaowen/GitHub/t2t/t2t_data/vocab.chn')
    encoder = ChineseIndexEncoderPickleSubword('/home/xuhaowen/GitHub/t2t/t2t_data/vocab_subword.chn')
    #encoder1 = EnglishIndexEncoderPickleConll('/home/xuhaowen/GitHub/t2t/t2t_data/vocab_conll.chn')
    #print(encoder.encode('，。\n二、2004年8月2日23时许，被告人金浩成伙同“军长”（在逃）在北京市朝阳区世纪村东区，盗走方晓光（男，42岁）停放于此的黑色本田雅阁牌轿车1辆（车牌号：京FJ7810，价值人民币20.2万元），后销赃。'))
    #print(encoder.encode('洇'))
    #print(encoder1.decode([0,1,2,3,4,5,6]))
    print(encoder1.encode('The outcome of the November elections emerged as a hot topic on Wall Street'))
