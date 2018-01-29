# coding:utf-8

import tensorflow as tf
import sys, os
father_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(father_dir)
from utils.generate_utils import isCapital

def chinese_generator(source_path, target_path, source_vocab, target_vocab, EOS=None, one_hot_feature=False):
    eos_list = [] if EOS is None else [EOS]
    use_one_hot = one_hot_feature

    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            while source and target:
                source_ints = source_vocab.encode(source.strip()) + eos_list
                target_ints = target_vocab.encode(target.strip()) + eos_list
                if use_one_hot:
                    one_hot_ints = [isCapital(word) for word in source.strip().split()] + eos_list
                    source_ints.extend(one_hot_ints)
                yield {"inputs": source_ints, "targets": target_ints}
                source, target = source_file.readline(), target_file.readline()
