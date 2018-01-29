# coding=utf-8
import os, sys
father_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(father_dir)
from config.path_config import *
from utils.label_translate import label_translate
from utils.label_translate import LabelFormat
sys.path.append(subword_nmt_path)
import learn_bpe
import codecs
import re
import shutil
import copy
from jpype import *
from data_generators import text_encoder
from collections import defaultdict

def build_subword_vocab(operation_num=100):
    infile = os.path.join(datagen_path, 'word_cut')
    outfile = infile + '_subword'
    learn_bpe.main(codecs.open(infile, encoding='utf-8'),
                   codecs.open(outfile, 'w', encoding='utf-8'),
                   operation_num, 2, verbose=True, vocab_name=vocab_file)
    print('run into here')

def word_segmentation_file(infile, outfile):
    hanlp_java_lib = os.path.join(hanlp_path, 'hanlp-1.3.2.jar')
    startJVM(getDefaultJVMPath(), "-Djava.class.path=%s:%s" %
             (hanlp_java_lib, hanlp_path), "-Xms1g", "-Xmx1g")
    HanLP = JClass('com.hankcs.hanlp.HanLP')
    NShortSegment = JClass('com.hankcs.hanlp.seg.NShort.NShortSegment')

    instream = open(infile, 'r')
    outstream = open(outfile, 'w')
    for line in instream.readlines():
        cut = HanLP.segment(line.strip())
        cut = [str(word).split('/')[0] for word in cut]
        cut = ' '.join(cut)
        outstream.write(cut+'\n')
    shutdownJVM()

def _iobe_consistency(seq):
    seq = [char != 'O' for char in seq]
    count = sum(seq)
    if count > len(seq) / 2:
        return '1'
    else:
        return '0'
    """
    if count == len(seq):
        return '1'
    elif count == 0:
        return '0'
    else:
        print(seq)
        print('inconstency error')
        exit()
    """

def subword_segmentation_file(in_word, out_word, in_iobe=None, out_iobe=None):
    encoder = text_encoder.ChineseIndexEncoderPickle(vocab_file)
    if in_iobe is None:
        in_iobe = in_word
    in_word_stream = open(in_word, 'r')
    in_iobe_stream = open(in_iobe, 'r')
    out_word_stream = open(out_word, 'w')
    if out_iobe is not None:
        out_iobe_stream = open(out_iobe, 'w')
    max_len_subword = max(len(subword) for subword in encoder._token_to_id.keys())
    vocab = encoder._token_to_id
    for line, line_iobe in zip(in_word_stream.readlines(), in_iobe_stream.readlines()):
        start = 0
        line = line.strip()
        line_iobe = line_iobe.strip()
        len_line = len(line)
        words = []
        iobe = []
        while start < len_line:
            for stop in range(start + max_len_subword, start + 1, -1):
                if stop >= len_line:
                    continue
                subword = line[start : stop]
                subword_iobe = line_iobe[start : stop]
                # subword founded
                if subword in vocab.keys():
                    words.append(subword)
                    iobe.append(_iobe_consistency(subword_iobe))
                    start = stop
                    break
            # not found
            subword = line[start]
            subword_iobe = line_iobe[start]
            words.append(subword)
            iobe.append(_iobe_consistency(subword_iobe))
            start = start + 1
        out_word_stream.write(' '.join(words) + '\n')
        if out_iobe is not None:
            out_iobe_stream.write(' '.join(iobe) + '\n')

def subword():
    tag = '_train'
    in_word = os.path.join(datagen_path, 'word%s' % tag)
    out_word = os.path.join(datagen_path, 'word_subword%s' % tag)
    in_iobe = os.path.join(datagen_path, 'iobe%s' % tag)
    out_iobe = os.path.join(datagen_path, 'iobe_subword%s' % tag)
    subword_segmentation_file(in_word, out_word, in_iobe=in_iobe, out_iobe=out_iobe)
    tag = '_valid'
    in_word = os.path.join(datagen_path, 'word%s' % tag)
    out_word = os.path.join(datagen_path, 'word_subword%s' % tag)
    in_iobe = os.path.join(datagen_path, 'iobe%s' % tag)
    out_iobe = os.path.join(datagen_path, 'iobe_subword%s' % tag)
    subword_segmentation_file(in_word, out_word, in_iobe=in_iobe, out_iobe=out_iobe)
    tag = '_test'
    in_word = os.path.join(datagen_path, 'word%s' % tag)
    out_word = os.path.join(datagen_path, 'word_subword%s' % tag)
    in_iobe = os.path.join(datagen_path, 'iobe%s' % tag)
    out_iobe = os.path.join(datagen_path, 'iobe_subword%s' % tag)
    subword_segmentation_file(in_word, out_word, in_iobe=in_iobe, out_iobe=out_iobe)

def cut():
    infile = os.path.join(datagen_path, 'word')
    outfile = os.path.join(datagen_path, 'word_cut')
    word_segmentation_file(infile, outfile)

def remove_long_case(tag, max_len=256):
    text_file = os.path.join(datagen_path, 'word_subword_%s' % tag)
    label_file = os.path.join(datagen_path, 'ner_subword_%s' % tag)
    text_stream = open(text_file, 'r')
    label_stream = open(label_file, 'r')
    texts = []
    labels = []
    for text, label in zip(text_stream.readlines(), label_stream.readlines()):
        if len(text.strip().split()) < max_len:
            texts.append(text)
            labels.append(label)
        else:
            print(text)
    text_stream.close()
    label_stream.close()
    text_stream = open(text_file, 'w')
    label_stream = open(label_file, 'w')
    for text, label in zip(texts, labels):
        text_stream.write(text)
        label_stream.write(label)
    text_stream.close()
    label_stream.close()

def copy_test():
    source = os.path.join(datagen_path, 'word_subword_test')
    target = os.path.join(data_path, 'decode_this_event_subword_full.txt')
    shutil.copy(source, target)
    source = os.path.join(datagen_path, 'ner_subword_iobe_test')
    target = os.path.join(data_path, 'target_this_event_subword_full.txt')
    shutil.copy(source, target)

def io2iobe():
    io_dict = {'I':'1', 'O':'0'}
    tag = 'test'
    io_file = os.path.join(datagen_path, 'ner_subword_%s' % tag)
    iobe_file = os.path.join(datagen_path, 'ner_subword_iobe_%s' % tag)
    label_translate(io_file, iobe_file, LabelFormat.IO, LabelFormat.IOBE,
                    source_dict=io_dict)

    tag = 'train'
    io_file = os.path.join(datagen_path, 'ner_subword_%s' % tag)
    iobe_file = os.path.join(datagen_path, 'ner_subword_iobe_%s' % tag)
    label_translate(io_file, iobe_file, LabelFormat.IO, LabelFormat.IOBE,
                    source_dict=io_dict)

    tag = 'valid'
    io_file = os.path.join(datagen_path, 'ner_subword_%s' % tag)
    iobe_file = os.path.join(datagen_path, 'ner_subword_iobe_%s' % tag)
    label_translate(io_file, iobe_file, LabelFormat.IO, LabelFormat.IOBE,
                    source_dict=io_dict)


if __name__ == '__main__':
    # word segmentation
    # cut()
    # build vocab
    build_subword_vocab(operation_num=0)
    # prepare subword train, valid and test data
    subword()
    # remove events longer than 256
    #remove_long_case('test', max_len=256)
    #remove_long_case('train', max_len=256)
    #remove_long_case('valid', max_len=256)
    # io tag to iobe tag
    io2iobe()
    # copy test file to data_path
    copy_test()
