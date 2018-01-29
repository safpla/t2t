# coding=utf-8
import os

root_path = '/home/xuhaowen/GitHub/t2t/'
source_path = '/home/xuhaowen/GitHub/t2t/deeplycurious'
datagen_path = '/home/xuhaowen/GitHub/t2t/t2t_datagen_event'
data_path = '/home/xuhaowen/GitHub/t2t/t2t_data'
train_path = '/home/xuhaowen/GitHub/t2t/t2t_train'
subword_nmt_path = '/home/xuhaowen/GitHub/subword-nmt'
oonp_project_path = '/home/xuhaowen/GitHub/oonp_source/oonp_judge'
raw_data_path = '/home/xuhaowen/GitHub/t2t/Data'
hanlp_path = '/home/xuhaowen/tools/hanlp/'
#vocab_file = os.path.join(data_path, 'vocab_subword.chn')
vocab_file = os.path.join(raw_data_path, 'pre_trained_word_emb_xxy/dict_pretrain.pkl')
pretrained_emb = os.path.join(raw_data_path, 'pre_trained_word_emb_xxy/embedding_matrix.pkl')
