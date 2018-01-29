# coding:utf-8
import os, sys
import codecs
import re
father_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(father_dir)
from config.path_config_msra import *
sys.path.append(subword_nmt_path)
import learn_bpe
from extract_to_txt import split_train_valid

def to_standard_input(input_file, word_file, ner_file):
    input_stream = codecs.open(input_file, 'r', 'utf-8')
    word_stream = codecs.open(word_file, 'w', 'utf-8')
    ner_stream = codecs.open(ner_file, 'w', 'utf-8')
    arabic_num = ['0','1','2','3','4','5','6','7','8','9']
    punctuation = ['。', '！']
    for line in input_stream.readlines():
        word, ner = line.strip().split()
        if word in arabic_num:
            word = '0'
        word_stream.write(word)
        ner_stream.write(ner)
        if word in punctuation:
            word_stream.write('\n')
            ner_stream.write('\n')
        else:
            word_stream.write(' ')
            ner_stream.write(' ')
    word_stream.close()
    ner_stream.close()

def extract():
    train_file = os.path.join(raw_data_path, 'msra_bakeoff3',
                              'msra_bakeoff3_training_char_label')
    word_file = os.path.join(datagen_path, 'word')
    ner_file = os.path.join(datagen_path, 'ner')
    to_standard_input(train_file, word_file, ner_file)

    ratio = 0.05
    split_train_valid(word_file, ner_file, ratio)

    test_file = os.path.join(raw_data_path, 'msra_bakeoff3',
                             'msra_bakeoff3_test_char_label')
    word_file = os.path.join(datagen_path, 'word_test')
    ner_file = os.path.join(datagen_path, 'ner_test')
    to_standard_input(test_file, word_file, ner_file)

def build_subword_vocab(operation_num=100):
    infile = os.path.join(datagen_path, 'word')
    outfile = infile + '_subword'
    learn_bpe.main(codecs.open(infile, encoding='utf-8'),
                   codecs.open(outfile, 'w', encoding='utf-8'),
                   operation_num, 2, verbose=True, vocab_name=vocab_file)


if __name__ == '__main__':
    extract()
    build_subword_vocab(operation_num=0)
