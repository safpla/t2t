from __future__ import absolute_import
import tensorflow as tf
import os
import sys
father_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(father_dir)
from config.path_config import *
from google.protobuf.json_format import MessageToJson
from data_generators import text_encoder

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def write_tfrecord(tfrecord):
    writer = tf.python_io.TFRecordWriter(tfrecord)

    labels = [[1,1,1,1,0,0,0,0],[1,2,3,4,5,6,7,8]]
    datas = [[1,2,43,1,5,55,1,1],[3,3,5,2,6,2,4,5]]
    for label, data in zip(labels, datas):
        feature = {'inputs': _int64_feature(label),
                   'targets': _int64_feature(data)}
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())

    writer.close()

def read_tfrecord(tfrecord):
    with tf.Session() as sess:
        feature = {'inputs': tf.VarLenFeature(tf.int64),
                   'targets': tf.VarLenFeature(tf.int64)}
        #feature = {'inputs': tf.FixedLenFeature([8], tf.int64),
        #           'targets': tf.FixedLenFeature([8], tf.int64)}
        filename_queue = tf.train.string_input_producer([tfrecord], num_epochs=1)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features=feature)
        label = features['targets']
        data = features['inputs']
        #label = tf.cast(features['targets'], tf.int64)
        #data = tf.cast(features['inputs'], tf.int64)
        #init = tf.global_variables_initializer()
        #sess.run(init)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(5):
            l, d = sess.run([label, data])
            print('label: %s, data:%s ' % (l, d))

def read_tfrecord_new(tfrecord, word_file, label_file, max_extract_num=None):
    word_stream = open(word_file, 'w')
    label_stream = open(label_file, 'w')
    input_encoder = text_encoder.ChineseIndexEncoderPickle(vocab_file)
    target_encoder = text_encoder.IOBELabelEncoderSubword()
    #input_encoder = text_encoder.EnglishIndexEncoderPickleConll(vocab_file)
    #target_encoder = text_encoder.LabelEncoderConll()
    count = 0
    for example in tf.python_io.tf_record_iterator(tfrecord):
        result = tf.train.Example.FromString(example)
        jsonMessage = MessageToJson(result)
        inputs = result.features.feature['inputs'].int64_list.value
        targets = result.features.feature['targets'].int64_list.value
        inputs = input_encoder.decode_list(inputs)
        targets = target_encoder.decode_list(targets)        # remove the last EOS character
        inputs = ' '.join(inputs[:-1])        # remove the last EOS character
        targets = ' '.join(targets[:-1])
        word_stream.write(inputs + '\n')
        label_stream.write(targets + '\n')
        count += 1
        if max_extract_num is not None and count >= max_extract_num:
            break
    word_stream.close()
    label_stream.close()


if __name__ == '__main__':
    folder_name = data_path
    tfrecord = '%s/word2ner_subword-train-00000-of-00001' % folder_name
    #tfrecord = '/home/xuhaowen/GitHub/Tagger/resources/ner/ner-train-00000-of-00001'
    word_file_full = '%s/decode_this.txt' % folder_name
    label_file_full = '%s/target_this.txt' % folder_name
    read_tfrecord_new(tfrecord, word_file_full, label_file_full, max_extract_num=10)
    #read_tfrecord(tfrecord)
