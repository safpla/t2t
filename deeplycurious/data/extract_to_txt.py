# coding:utf-8
import os
import sys
import random
import pickle as pkl
import numpy as np

father_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(father_dir)
from config.path_config import *
sys.path.append(oonp_project_path)

from dataio import dataio

#datagen_path = os.path.join(root_path, 't2t_datagen_sentence')
def _no_label(seq):
    seq = [char == 'O' for char in seq]
    return True if sum(seq) == len(seq) else False

def extract_npy(text_file, label_file, word_file, ner_file, granularity='event'):
    word_stream = open(word_file, 'w')
    ner_stream = open(ner_file, 'w')
    word = np.load(text_file, encoding='bytes')
    ner = np.load(label_file, encoding='bytes')
    label_dict = {0:'O', 1:'B', 2:'I', 3:'E', 4:'S'}
    for case_w, case_n in zip(word, ner):
        for event_w, event_n in zip(case_w, case_n):
            event_n = [label_dict[c] for c in event_n]
            event_w = event_w[0:-1]
            event_n = event_n[0:-1]
            if _no_label(event_n):
                continue
            for c_w, c_n in zip(event_w, event_n):
                if c_w == '\n' or c_w == ' ':
                    pass
                elif c_w == '。' and granularity == 'sentence':
                    word_stream.write(c_w)
                    word_stream.write('\n')
                    ner_stream.write(c_n)
                    ner_stream.write('\n')
                else:
                    word_stream.write(c_w)
                    ner_stream.write(c_n)
            word_stream.write('\n')
            ner_stream.write('\n')
            if len(event_w) != len(event_n):
                print('error: different text and label length')

def extract_pkl(feature_pkl, target_pkl, feature_txt, target_txt, granularity='event'):
    with open(feature_pkl, 'rb') as f:
        features = pkl.load(f)
    with open(target_pkl, 'rb') as f:
        targets = pkl.load(f)

    label_dict = {0:'O', 1:'B', 2:'I', 3:'E'}
    with open(feature_txt, 'w') as feature_out:
        with open(target_txt, 'w') as target_out:
            for feature, target in zip(features, targets):
                for ind, word in enumerate(feature):
                    if word == '\n':
                        feature[ind] = '。'
                target = [label_dict[i] for i in target]
                feature_out.write(' '.join(feature))
                feature_out.write('\n')
                target_out.write(' '.join(target))
                target_out.write('\n')


def extract_json(granularity='event'):
    """
    granularity: string, if is 'event', use event as granularity
    if is 'sentence', use sentence as granularity
    """
    data_file = os.path.join(oonp_project_path, 'data/json.2017.10.10')
    #data_file = '../../Data/json.train'
    datas, labels = dataio.entity_people_process(data_file)
    labels = labels[0]
    data_path = os.path.join(root_path, 't2t_datagen_%s' % granularity)
    word_train_file = os.path.join(data_path, 'word_train')
    word_valid_file = os.path.join(data_path, 'word_valid')
    ner_train_file = os.path.join(data_path, 'ner_train')
    ner_valid_file = os.path.join(data_path, 'ner_valid')
    word_stream_train = open(word_train_file, 'w')
    word_stream_valid = open(word_valid_file, 'w')
    ner_stream_train = open(ner_train_file, 'w')
    ner_stream_valid = open(ner_valid_file, 'w')
    label_dict = {0:'O', 1:'B', 2:'I', 3:'E', 4:'S'}
    ratio = 0.1
    for data, label in zip(datas, labels):
        r = random.random()
        if r < ratio:
            word_stream = word_stream_valid
            ner_stream = ner_stream_valid
        else:
            word_stream = word_stream_train
            ner_stream = ner_stream_train
        for char, ind in zip(data, label):
            if char is '\n':
                if granularity != 'sentence':
                    word_stream.write('\n')
                    ner_stream.write('\n')
            elif (char == '。' or char == '；') and granularity == 'sentence':
                word_stream.write(char)
                word_stream.write('\n')
                ner_stream.write(label_dict[ind])
                ner_stream.write('\n')
            elif char is ' ':
                pass
            else:
                word_stream.write(char)
                ner_stream.write(label_dict[ind])
        if data[-1] != '\n':
            word_stream.write('\n')
            ner_stream.write('\n')

    word_stream.close()
    ner_stream.close()

def split_train_valid(text_file, label_file, ratio):
    text_train = text_file + '_train'
    text_valid = text_file + '_valid'
    label_train = label_file + '_train'
    label_valid = label_file + '_valid'

    word_file_stream = open(text_file, 'r')
    ner_file_stream = open(label_file, 'r')
    word_train_stream = open(text_train, 'w')
    word_valid_stream = open(text_valid, 'w')
    ner_train_stream = open(label_train, 'w')
    ner_valid_stream = open(label_valid, 'w')
    max_len = 0
    count = 0
    word_files = word_file_stream.readlines()
    ner_files = ner_file_stream.readlines()
    total_file_num = len(word_files)
    for data ,label in zip(word_files, ner_files):
        if data.strip() == '':
            continue
        max_len = max(max_len, len(data))
        count += 1
        r = count / total_file_num
        if r < ratio:
            word_valid_stream.write(data)
            ner_valid_stream.write(label)
        else:
            word_train_stream.write(data)
            ner_train_stream.write(label)
    word_train_stream.close()
    word_valid_stream.close()
    ner_train_stream.close()
    ner_valid_stream.close()
    word_file_stream.close()
    ner_file_stream.close()
    #os.remove(text_file)
    #os.remove(label_file)

def split_by_space(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    with open(filename, 'w') as f:
        for line in lines:
            f.write(' '.join([c for c in line]))

def testfy():
    word_file = 'word'
    ner_file = 'ner'
    word_stream = open('word', 'r')
    ner_stream = open('ner', 'r')
    for data ,label in zip (word_stream.readlines(), ner_stream.readlines()):
        print(data)
        print(label)
        print(len(data))
        print(len(label))


if __name__ == '__main__':
    # extract pickle
    rawdata_path = os.path.join(root_path, 'Data/save_data')
    feature_pkl = os.path.join(rawdata_path, 'train_data.pkl')
    target_pkl = os.path.join(rawdata_path, 'train_label.pkl')
    feature_txt = os.path.join(datagen_path, 'word_train')
    target_txt = os.path.join(datagen_path, 'iobe_train')
    extract_pkl(feature_pkl, target_pkl, feature_txt, target_txt)

    feature_pkl = os.path.join(rawdata_path, 'test_data.pkl')
    target_pkl = os.path.join(rawdata_path, 'test_label.pkl')
    feature_txt = os.path.join(datagen_path, 'word_test')
    target_txt = os.path.join(datagen_path, 'iobe_test')
    extract_pkl(feature_pkl, target_pkl, feature_txt, target_txt)

    feature_pkl = os.path.join(rawdata_path, 'valid_data.pkl')
    target_pkl = os.path.join(rawdata_path, 'valid_label.pkl')
    feature_txt = os.path.join(datagen_path, 'word_valid')
    target_txt = os.path.join(datagen_path, 'iobe_valid')
    extract_pkl(feature_pkl, target_pkl, feature_txt, target_txt)
    exit()

    #extract_json()
    rawdata_path = os.path.join(root_path, 'Data/npy')

    text_file_in = os.path.join(rawdata_path, 'test_case.npy')
    label_file_in = os.path.join(rawdata_path, 'test_bmeo.npy')
    text_file_out = os.path.join(datagen_path, 'word_test')
    label_file_out = os.path.join(datagen_path, 'iobe_test')
    extract_npy(text_file_in, label_file_in, text_file_out, label_file_out,
                granularity='event')
    split_by_space(text_file_out)
    split_by_space(label_file_out)

    text_file_in = os.path.join(rawdata_path, 'train_case.npy')
    label_file_in = os.path.join(rawdata_path, 'train_bmeo.npy')
    text_file_out = os.path.join(datagen_path, 'word_train')
    label_file_out = os.path.join(datagen_path, 'iobe_train')
    extract_npy(text_file_in, label_file_in, text_file_out, label_file_out,
                granularity='event')
    split_by_space(text_file_out)
    split_by_space(label_file_out)

    rawdata_path = os.path.join(root_path, 'Data/npy')
    text_file_in = os.path.join(rawdata_path, 'valid_case.npy')
    label_file_in = os.path.join(rawdata_path, 'valid_bmeo.npy')
    text_file_out = os.path.join(datagen_path, 'word_valid')
    label_file_out = os.path.join(datagen_path, 'iobe_valid')
    extract_npy(text_file_in, label_file_in, text_file_out, label_file_out,
                granularity='event')
    split_by_space(text_file_out)
    split_by_space(label_file_out)
    # ratio = 0.1
    # split_train_valid(text_file_out, label_file_out, ratio)
    # text_file_in = os.path.join(rawdata_path, 'test_case.npy')
    # label_file_in = os.path.join(rawdata_path, 'test_bmeo.npy')
    # text_file_out = os.path.join(datagen_path, 'word_test')
    # label_file_out = os.path.join(datagen_path, 'ner_test')
    # extract_npy(text_file_in, label_file_in, text_file_out, label_file_out,
    #             granularity='event')
