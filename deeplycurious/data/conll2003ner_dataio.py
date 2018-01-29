from collections import OrderedDict
import os, sys
import pickle as pkl
import codecs
import csv
import itertools
from functools import reduce
father_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(father_dir)
from config.path_config_conll import *
import numpy as np
import shutil
import torch
import json
import re
import torch.nn as nn
import torch.nn.init

def isCapital(word):
    if re.match('^[A-Z]+', word):
        return 0
    elif re.match('^[a-zA-Z]+', word):
        return 1
    elif re.match('^[0-9a-zA-Z]+',word):
        return 2
    else:
        return 3

def generate_corpus(lines, if_shrink_feature=False, thresholds=1):
    """
    generate label, feature, word dictionary and label dictionary

    args:
        lines : corpus
        if_shrink_feature: whether shrink word-dictionary
        threshold: threshold for shrinking word-dictionary

    """
    features = list()
    features_cap = list()
    labels = list()
    tmp_fl = list()
    tmp_Bigl = list()
    tmp_ll = list()
    feature_map = dict()
    label_map = dict()
    for line in lines:
        if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
            line = line.rstrip('\n').split()
            word = line[0]# .lower()
            tmp_fl.append(word.strip().strip('.'))
            # tmp_fl.append(word)
            tmp_Bigl.append(isCapital(line[0]))
            if word not in feature_map:
                feature_map[word] = len(feature_map) + 1 #0 is for unk
            tmp_ll.append(line[-1])
            if line[-1] not in label_map:
                label_map[line[-1]] = len(label_map)
        elif len(tmp_fl) > 0:
            features.append(tmp_fl)
            features_cap.append(tmp_Bigl)
            labels.append(tmp_ll)
            tmp_fl = list()
            tmp_Bigl = list()
            tmp_ll = list()
    if len(tmp_fl) > 0:
        features.append(tmp_fl)
        features_cap.append(tmp_Bigl)
        labels.append(tmp_ll)
    label_map['<start>'] = len(label_map)
    label_map['<pad>'] = len(label_map)
    if if_shrink_feature:
        feature_map = shrink_features(feature_map, features, thresholds)
    else:
        #inserting unk to be 0 encoded
        feature_map['<unk>'] = 0
        #inserting eof
        feature_map['<eof>'] = len(feature_map)

    return features, [labels, features_cap], feature_map, label_map

def generate_corpus_char(lines, if_shrink_c_feature=False, c_thresholds=1, if_shrink_w_feature=False, w_thresholds=1):
    """
    generate label, feature, word dictionary, char dictionary and label dictionary

    args:
        lines : corpus
        if_shrink_c_feature: whether shrink char-dictionary
        c_threshold: threshold for shrinking char-dictionary
        if_shrink_w_feature: whether shrink word-dictionary
        w_threshold: threshold for shrinking word-dictionary

    """
    features, labels, feature_map, label_map = generate_corpus(lines, if_shrink_feature=if_shrink_w_feature, thresholds=w_thresholds)
    char_count = dict()
    for feature in features:
        for word in feature:
            for tup in word:
                if tup not in char_count:
                    char_count[tup] = 0
                else:
                    char_count[tup] += 1
    if if_shrink_c_feature:
        shrink_char_count = [k for (k, v) in iter(char_count.items()) if v >= c_thresholds]
        char_map = {shrink_char_count[ind]: ind for ind in range(0, len(shrink_char_count))}
    else:
        char_map = {k: v for (v, k) in enumerate(char_count.keys())}
    char_map['<u>'] = len(char_map)  # unk for char
    char_map[' '] = len(char_map)  # concat for char
    char_map['\n'] = len(char_map)  # eof for char
    return features, labels, feature_map, label_map, char_map

def read_corpus(lines):
    """
    convert corpus into features and labels
    """
    features = list()
    labels = list()
    tmp_fl = list()
    tmp_ll = list()
    for line in lines:
        if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
            line = line.rstrip('\n').split()
            tmp_fl.append(line[0])
            tmp_ll.append(line[-1])
        elif len(tmp_fl) > 0:
            features.append(tmp_fl)
            labels.append(tmp_ll)
            tmp_fl = list()
            tmp_ll = list()
    if len(tmp_fl) > 0:
        features.append(tmp_fl)
        labels.append(tmp_ll)

    return features, labels

def shrink_features(feature_map, features, thresholds):
    """
    filter un-common features by threshold
    """
    feature_count = {k: 0 for (k, v) in iter(feature_map.items())}
    for feature_list in features:
        for feature in feature_list:
            feature_count[feature] += 1
    shrinked_feature_count = [k for (k, v) in iter(feature_count.items()) if v >= thresholds]
    feature_map = {shrinked_feature_count[ind]: (ind + 1) for ind in range(0, len(shrinked_feature_count))}

    #inserting unk to be 0 encoded
    feature_map['<mask>'] =  0
    feature_map['<unk>'] =  len(feature_map)
    #inserting eof
    feature_map['<eof>'] = len(feature_map)
    return feature_map

class Dataset_conll(object):

    def __init__(self,  rnd_seed=1234, vocab = None,minicount = 5, vocab_vec=None):
        self.train_path = 'data/eng.train'
        self.valid_path = 'data/eng.testa'
        self.test_path = 'data/eng.testb'
        self.rnd_seed = rnd_seed
        print('loading corpus......')
        docs_all, labels_all, self.label_map = self.readfile(self.train_path)
        random_order = np.random.RandomState(rnd_seed).permutation(len(docs_all))
        docs_all = [docs_all[i] for i in random_order]
        labels_all = [[label_all[i] for i in random_order] for label_all in labels_all]
        self.train_docs = docs_all
        self.test_docs, test_labels, _ = self.readfile(self.test_path)
        self.valid_docs, valid_labels,_ = self.readfile(self.valid_path)

        idex = 0
        if vocab_vec != None: # vocab_vec contains unk, mask
            self.vocab = dict()
            for word in vocab.keys():
                if vocab_vec.has_key(word):
                    self.vocab[word] = idex
                    idex += 1
            del vocab_vec
        elif vocab != None:
            self.vocab = vocab
        else:
            self.vocab = self.genVocab(docs_all, minicount)

        index_docs = self.word2index(self, self.train_docs, self.vocab)
        self.train_data = index_docs
        valid_index_docs = self.word2index(self, self.valid_docs, self.vocab)
        self.valid_data = valid_index_docs
        test_index_docs = self.word2index(self, self.test_docs, self.vocab)
        self.test_data = test_index_docs

        self.train_labels = [self.word2index(self, labels_all[0], self.label_map), labels_all[1]]
        print('loaded data {} labels {}'.format(len(self.train_data), len(self.train_labels[0])))

        self.valid_labels = [self.word2index(self, valid_labels[0], self.label_map),valid_labels[1]]
        self.test_labels = [self.word2index(self, test_labels[0], self.label_map), test_labels[1]]

    def readfile(self, data_path):

        with codecs.open(data_path, 'r', 'utf-8') as f:
            lines = f.readlines()
        train_features, train_labels, vocab, label_map =\
        generate_corpus(lines,if_shrink_feature=False, thresholds=0)

        return train_features, train_labels, label_map

    def genVocab(self,lines,minicount, maskid=0):
        """generate vocabulary from contents"""
        #lines = [' '.join(line) for line in lines]
        wordset = set(item for line in lines for item in line)
        freq = {word: 0 for index, word in enumerate(wordset)}
        for line in lines:
            line_wordset = set(item for item in line)
            for word in line_wordset:
                freq[word] += 1
        high_word = []
        for (word,count) in freq.items():
            if count >minicount and word != 'mask':
                high_word.append(word)

        word2index = {word: index + 1 for index, word in enumerate(high_word)}
        word2index['<mask>'] = maskid
        word2index['<unk>'] = len(word2index)

        return word2index


    def getIndex(self,word, vocab):
        if word in vocab:
            return vocab[word]
        else:
            return vocab['<unk>']

    @staticmethod
    def word2index(self, docs, vocab):
        index_docs = [[self.getIndex(char, vocab) for char in doc] for doc in docs]
        return index_docs

    def getData(self):
        return self.train_data, self.train_labels, self.valid_data, self.valid_labels,\
            self.test_data, self.test_labels

    def getVocab(self):
        return self.vocab, self.label_map

    def getBowf(self):
        # bow feature
        self.bowdata = []
        for dt in self.train_data:
            sample = [0]*len(self.vocab)
            for dt_id  in dt:
                sample[dt_id] += 1
            self.bowdata.append(sample)
        self.labels = self.train_labels

        self.valid_bowdata = []
        for dt in self.valid_data:
            sample = [0]*len(self.vocab)
            for dt_id  in dt:
                sample[dt_id] += 1
            self.valid_bowdata.append(sample)

        self.test_bowdata = []
        for dt in self.test_data:
            sample = [0]*len(self.vocab)
            for dt_id  in dt:
                sample[dt_id] += 1
            self.test_bowdata.append(sample)

        print('loaded vocab size {} '.format(len(self.vocab)))
        print('loaded training data {} label {}'.format(len(self.bowdata), self.labels.shape))
        return self.bowdata, self.labels, self.valid_bowdata,  self.valid_labels,\
                self.test_bowdata, self.test_labels

def test():
    # load corpus
    train_file = os.path.join(raw_data_path, 'conll2003ner/eng.train')
    dev_file = os.path.join(raw_data_path, 'conll2003ner/eng.testa')
    test_file = os.path.join(raw_data_path, 'conll2003ner/eng.testb')
    print('loading corpus')
    with codecs.open(train_file, 'r', 'utf-8') as f:
        lines = f.readlines()
    with codecs.open(dev_file, 'r', 'utf-8') as f:
        dev_lines = f.readlines()
    with codecs.open(test_file, 'r', 'utf-8') as f:
        test_lines = f.readlines()

    dev_features, dev_labels = read_corpus(dev_lines)
    test_features, test_labels = read_corpus(test_lines)
    mini_count = 3
        # converting format
    train_features, train_labels, f_map, l_map, c_map = generate_corpus_char(lines, if_shrink_c_feature=True, c_thresholds=mini_count, if_shrink_w_feature=False)
    tag = '_train'
    out_features = os.path.join(datagen_path, 'word%s' % tag)
    out_labels = os.path.join(datagen_path, 'ner%s' % tag)
    features_stream = open(out_features, 'w')
    labels_stream = open(out_labels, 'w')
    for features, labels in zip(train_features, train_labels[0]):
        new_features = []
        new_labels = []
        for feature, label in zip(features, labels):
            if feature != '':
                new_features.append(feature)
                new_labels.append(label)
        features_stream.write(' '.join(new_features) + '\n')
        labels_stream.write(' '.join(new_labels) + '\n')
    features_stream.close()
    labels_stream.close()
    # save vocab
    d = OrderedDict()
    count = 1
    for word, ind in f_map.items():
        count += 1
        d[word] = count

    dev_features, dev_labels, f_map, l_map, c_map = generate_corpus_char(dev_lines, if_shrink_c_feature=True, c_thresholds=mini_count, if_shrink_w_feature=False)
    tag = '_valid'
    out_features = os.path.join(datagen_path, 'word%s' % tag)
    out_labels = os.path.join(datagen_path, 'ner%s' % tag)
    features_stream = open(out_features, 'w')
    labels_stream = open(out_labels, 'w')
    for features, labels in zip(dev_features, dev_labels[0]):
        new_features = []
        new_labels = []
        for feature, label in zip(features, labels):
            if feature != '':
                new_features.append(feature)
                new_labels.append(label)
        features_stream.write(' '.join(new_features) + '\n')
        labels_stream.write(' '.join(new_labels) + '\n')
    features_stream.close()
    labels_stream.close()
    # save vocab
    for word, ind in f_map.items():
        if word not in d:
            count += 1
            d[word] = count

    test_features, test_labels, f_map, l_map, c_map = generate_corpus_char(test_lines, if_shrink_c_feature=True, c_thresholds=mini_count, if_shrink_w_feature=False)
    tag = '_test'
    out_features = os.path.join(datagen_path, 'word%s' % tag)
    out_labels = os.path.join(datagen_path, 'ner%s' % tag)
    features_stream = open(out_features, 'w')
    labels_stream = open(out_labels, 'w')
    for features, labels in zip(test_features, test_labels[0]):
        new_features = []
        new_labels = []
        for feature, label in zip(features, labels):
            if feature != '':
                new_features.append(feature)
                new_labels.append(label)
        features_stream.write(' '.join(new_features) + '\n')
        labels_stream.write(' '.join(new_labels) + '\n')
    features_stream.close()
    labels_stream.close()
    """
    # save vocab
    for word, ind in f_map.items():
        if word not in d:
            count += 1
            d[word] = count
    """
    d = OrderedDict(sorted(d.items(), key=lambda t:t[1]))
    freqs = []
    dump = [d, freqs]
    f = open(vocab_file, 'wb')
    pkl.dump(dump, f, 2)
    f.close()
    # print(train_features)
    # print(train_labels[0])
    # print(train_labels[1])
    # print(f_map)  # vocab
    # print(l_map)  # label vocab
    # print(c_map)  # char vocab
# test()

if __name__ == '__main__':
    test()
