# coding:utf-8
import os, sys
father_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(father_dir)
from config.path_config import *
from utils.label_translate import label_translate, LabelFormat

def label_clean(source_file, target_file, reserve_label, replacement):
    source_stream = open(source_file, 'r')
    target_stream = open(target_file, 'w')
    for line in source_stream.readlines():
        line = line.strip().split()
        line_new = []
        for word in line:
            if word in reserve_label:
                line_new.append(word)
            else:
                line_new.append(replacement)
        target_stream.write(' '.join(line_new) + '\n')
    source_stream.close()
    target_stream.close()

def word_label_align(word_file, label_files, aligned_file, use_subword=False):
    empty = 'EMPTY'
    word_stream = open(word_file, 'r')
    label_streams = [open(label_file, 'r') for label_file in label_files]
    aligned_stream = open(aligned_file, 'w')
    word_line = word_stream.readlines()
    labels_line = [label_stream.readlines() for label_stream in label_streams]
    for i in range(len(word_line)):
        word = word_line[i].strip()
        if use_subword:
            word = word.split()
        else:
            word = [w for w in word]
        labels = [label_line[i].strip() for label_line in labels_line]
        for j in range(len(labels)):
            if use_subword:
                labels[j] = labels[j].split()
            else:
                labels[j] = [l for l in labels[j]]
        mess = pad_to_same_length([word] + labels, empty)
        for j in range(len(mess[0])):
            cell = '(%s' % mess[0][j]
            for k in range(1,len(mess)):
                cell = cell + '|%s' % mess[k][j]
            cell = cell + ')'
            if j == len(mess[0]) - 1:
                cell = cell + '\n'
            else:
                cell = cell + ' '
            aligned_stream.write(cell)
    aligned_stream.close()

def pad_to_same_length(ss, pad, l_max=None):
    """
    pad lists to the same length (the longest one if l_max==None,
    else l_max) with 'pad'
    """
    ls = []
    for s in ss:
        ls.append(len(s))
    if l_max == None:
        l_max = max(ls)
    ss_new = []
    for s, l in zip(ss, ls):
        if len(s) <= l_max:
            s.extend([pad] * (l_max - l))
        else:
            s = s[:l_max]
        ss_new.append(s)
    return ss_new

def compute_performance(label, pred_label, text=None):
    '''1：开头；2：中间；3：结束；0：其他'''

    def _parse(s):
        s = s.split('_')
        return int(s[0]), int(s[1])

    def _show_error_case(text, _begin, _end, _error_type):
        s = '%s: %s[%s]%s' % (_error_type, ''.join(text[max(0, _begin-10):_begin]),
                              ''.join(text[_begin:_end]),
                              ''.join(text[_end:min(len(text), _end+10)]))
        '''
        s = '%s: %s[%s]%s' % (_error_type, ''.join(text[0:_begin]),
                              ''.join(text[_begin:_end]),
                              ''.join(text[_end:-1]))
        '''
        print(s)

    def gen_entities(l):
        del_ixs = []
        entities = dict()
        if 1 in l:
            ixs, _ = zip(*filter(lambda x: x[1] == 1, enumerate(l)))
            ixs = list(ixs)
            ixs.append(len(label))
            for i in range(len(ixs) - 1):
                sub_label = l[ixs[i]:ixs[i + 1]]
                end_mark = max(sub_label)
                end_ix = ixs[i] + sub_label.index(end_mark) + 1
                entities['{}_{}'.format(ixs[i], end_ix)] = l[ixs[i]:end_ix]
                del_ixs.extend(range(ixs[i], end_ix))
        return entities, del_ixs

    if text != None:
        show_error = True
    else:
        show_error = False
    g_entities, _ = gen_entities(label)
    p_entities, del_ixs = gen_entities(pred_label)
    count_tp = 0
    count_fp = 0
    for p_t in p_entities.keys():
        _begin, _end = _parse(p_t)
        try:
            if p_entities[p_t] == g_entities[p_t]:
                count_tp += 1
            else:
                count_fp += 1
                if show_error:
                    _show_error_case(text, _begin, _end, 'false_positive')
        except:
            count_fp += 1
            if show_error:
                _show_error_case(text, _begin, _end, 'false_positive')

    count_tn = 0
    count_fn = 0

    for g_t in g_entities.keys():
        _begin, _end = _parse(g_t)
        try:
            if g_entities[g_t] == p_entities[g_t]:
                pass
            else:
                count_fn += 1
                if show_error:
                    _show_error_case(text, _begin, _end, 'missing')
        except:
            count_fn += 1
            if show_error:
                _show_error_case(text, _begin, _end, 'missing')

    return count_tp, count_fp, count_fn

def show_decoding_result(pred_file, gdth_file, text_file='', subword=False):
    """
    pred_file: prediction, each line is a sample
    gdth_file: groundtruth, each line is a sample
    text_file: raw text, if provided, error cases will be printed out
    """

    pred_stream = open(pred_file, 'r')
    gdth_stream = open(gdth_file, 'r')
    if text_file != '':
        text_stream = open(text_file, 'r')
    mapping = {'I': 2, 'O':0, 'B': 1, 'E':3, 'S': 1}
    count_tp = count_fp = count_fn = 0
    while 1:
        pred = pred_stream.readline()
        gdth = gdth_stream.readline()
        if pred == '' or gdth == '':
            break
        if subword:
            pred = pred.strip().split(' ')
            gdth = gdth.strip().split(' ')
        else:
            pred = pred.strip()
            gdth = gdth.strip()
        pred = [mapping[c] for c in pred]
        gdth = [mapping[c] for c in gdth]
        # pad the prediction to the length of ground truth
        [pred, gdth] = pad_to_same_length([pred, gdth], 0, len(gdth))

        if text_file != '':
            text = text_stream.readline()
            if subword:
                text = text.strip().split()
            else:
                text = [w for w in text.strip()]
        else:
            text = None
        tp, fp, fn = compute_performance(gdth, pred, text)
        count_tp += tp
        count_fp += fp
        count_fn += fn

    if count_tp + count_fp != 0:
        pcs = count_tp / float(count_tp + count_fp)
    else:
        pcs = None
    if count_tp + count_fn != 0:
        recall = count_tp / float(count_tp + count_fn)
    else:
        recall = None
    try:
        f1 = 2 * pcs * recall / float(pcs + recall)
    except:
        f1 = None
    print('precision: %s, recall: %s, f1: %s' %(pcs, recall, f1))
    return pcs, recall, f1, count_tp, count_fp, count_fn

def align():
    tag = 'event_subword_full'
    word_file = os.path.join(data_path, 'decode_this_%s.txt' % tag)
    label_file1 = os.path.join(data_path, 'decode_this_event_subword_full.txt.transformer_sl.word2ner_subword_hparams_long.word2ner_subword.beam1.alpha1.0.decodes')
    label_file1 = os.path.join(data_path, 'combined_test.deepatt.decodes')
    label_file2 = os.path.join(data_path, 'target_this_%s.txt' % tag)
    aligned_file = os.path.join(data_path, 'result.txt')
    #word_label_align(word_file, [label_file2], aligned_file, use_subword=True)
    word_label_align(word_file, [label_file1, label_file2], aligned_file, use_subword=True)
    exit()

    #from config.path_config_conll import datagen_path
    word_file = os.path.join(datagen_path, 'word_train')
    label_file1 = os.path.join(datagen_path, 'ner_train')
    label_file2 = os.path.join(datagen_path, 'word_test.transformer.word2ner_conll_hparams.word2ner_conll.beam4.alpha1.0.decodes')
    word_stream = open(word_file, 'r')
    label_stream = open(label_file1, 'r')
    for word, label in zip(word_stream.readlines(), label_stream.readlines()):
        word = word.split(' ')
        label = label.split(' ')
        if len(word) != len(label):
            print(word)
            print(len(word))
            print(label)
            print(len(label))
            print('\n')
    aligned_file = os.path.join(datagen_path, 'result.txt')
    word_label_align(word_file, [label_file1], aligned_file, use_subword=False)

def performance():
    folder_name = 't2t_data'
    num_samples = '_event_full'
    num_samples = '_event_subword_full'
    #pred_file = '/home/xuhaowen/GitHub/t2t/%s/decode_this%s.txt.transformer.word2ner_hparams.word2ner.beam4.alpha1.0.decodes' % (folder_name, num_samples)
    pred_file = '/home/xuhaowen/GitHub/t2t/%s/decode_this%s.txt.transformer.word2ner_hparams_thin.word2ner.beam4.alpha1.0.decodes' % (folder_name, num_samples)
    pred_file = '/home/xuhaowen/GitHub/t2t/%s/decode_this%s.txt.transformer.word2ner_subword_hparams_thin.word2ner_subword.beam4.alpha1.0.decodes' % (folder_name, num_samples)
    #pred_file = '/home/xuhaowen/GitHub/t2t/%s/decode_this%s.txt.transformer_sl.word2ner_subword_hparams_deep8.word2ner_subword.beam1.alpha1.0.decodes' % (folder_name, num_samples)
    gdth_file = '/home/xuhaowen/GitHub/t2t/%s/target_this%s.txt' % (folder_name, num_samples)
    text_file = '/home/xuhaowen/GitHub/t2t/%s/decode_this%s.txt' % (folder_name, num_samples)

    #pred_file = '/home/xuhaowen/GitHub/Tagger/combined_train.deepatt.decodes'
    #gdth_file = '/home/xuhaowen/GitHub/t2t/t2t_datagen_event/ner_subword_iobe_train'
    #text_file = '/home/xuhaowen/GitHub/t2t/t2t_datagen_event/word_subword_train'
    show_decoding_result(pred_file, gdth_file, text_file, subword=True)
    #show_decoding_result(pred_file, gdth_file, subword=True)
    #show_decoding_result(pred_file, gdth_file, subword=False)

def performance_conll():
    from config.path_config_conll import datagen_path
    #pred_file = os.path.join(datagen_path, 'word_test.transformer.word2ner_conll_hparams.word2ner_conll.beam4.alpha1.0.decodes')
    #gdth_file = os.path.join(datagen_path, 'ner_test')
    pred_file = os.path.join(datagen_path, 'word_test_onehotfeature.txt.transformer.word2ner_conll_hparams_singlehead.word2ner_conll.beam4.alpha1.0.decodes')
    gdth_file = os.path.join(datagen_path, 'word_test_onehotfeature.txt.transformer.word2ner_conll_hparams_singlehead.word2ner_conll.beam4.alpha1.0.targets')
    text_file = os.path.join(datagen_path, 'word_test')
    count_tp = 0
    count_fp = 0
    count_fn = 0

    pred_per = os.path.join(datagen_path, 'pred_per')
    gdth_per = os.path.join(datagen_path, 'gdth_per')
    pred_per_iobe = os.path.join(datagen_path, 'pred_per_iobe')
    gdth_per_iobe = os.path.join(datagen_path, 'gdth_per_iobe')
    label_clean(pred_file, pred_per, ['O', 'I-PER', 'B-PER'], 'O')
    label_clean(gdth_file, gdth_per, ['O', 'I-PER', 'B-PER'], 'O')
    iob_dict = {'O':'O', 'I':'I-PER', 'B':'B-PER'}
    label_translate(pred_per, pred_per_iobe, source_format=LabelFormat.IOB,
                    target_format=LabelFormat.IOBE, source_dict=iob_dict)
    label_translate(gdth_per, gdth_per_iobe, source_format=LabelFormat.IOB,
                    target_format=LabelFormat.IOBE, source_dict=iob_dict)
    _, _, _, tp, fp, fn = show_decoding_result(pred_per_iobe, gdth_per_iobe,
                                               subword=True)
    count_tp += tp
    count_fp += fp
    count_fn += fn

    pred_loc= os.path.join(datagen_path, 'pred_loc')
    gdth_loc = os.path.join(datagen_path, 'gdth_loc')
    pred_loc_iobe = os.path.join(datagen_path, 'pred_loc_iobe')
    gdth_loc_iobe = os.path.join(datagen_path, 'gdth_loc_iobe')
    label_clean(pred_file, pred_loc, ['O', 'I-LOC', 'B-LOC'], 'O')
    label_clean(gdth_file, gdth_loc, ['O', 'I-LOC', 'B-LOC'], 'O')
    iob_dict = {'O':'O', 'I':'I-LOC', 'B':'B-LOC'}
    label_translate(pred_loc, pred_loc_iobe, source_format=LabelFormat.IOB,
                    target_format=LabelFormat.IOBE, source_dict=iob_dict)
    label_translate(gdth_loc, gdth_loc_iobe, source_format=LabelFormat.IOB,
                    target_format=LabelFormat.IOBE, source_dict=iob_dict)
    _, _, _, tp, fp, fn = show_decoding_result(pred_loc_iobe, gdth_loc_iobe,
                                               subword=True)
    count_tp += tp
    count_fp += fp
    count_fn += fn

    pred_org = os.path.join(datagen_path, 'pred_org')
    gdth_org = os.path.join(datagen_path, 'gdth_org')
    pred_org_iobe = os.path.join(datagen_path, 'pred_org_iobe')
    gdth_org_iobe = os.path.join(datagen_path, 'gdth_org_iobe')
    label_clean(pred_file, pred_org, ['O', 'I-ORG', 'B-ORG'], 'O')
    label_clean(gdth_file, gdth_org, ['O', 'I-ORG', 'B-ORG'], 'O')
    iob_dict = {'O':'O', 'I':'I-ORG', 'B':'B-ORG'}
    label_translate(pred_org, pred_org_iobe, source_format=LabelFormat.IOB,
                    target_format=LabelFormat.IOBE, source_dict=iob_dict)
    label_translate(gdth_org, gdth_org_iobe, source_format=LabelFormat.IOB,
                    target_format=LabelFormat.IOBE, source_dict=iob_dict)
    _, _, _, tp, fp, fn = show_decoding_result(pred_org_iobe, gdth_org_iobe,
                                               subword=True)
    count_tp += tp
    count_fp += fp
    count_fn += fn

    pred_misc = os.path.join(datagen_path, 'pred_misc')
    gdth_misc = os.path.join(datagen_path, 'gdth_misc')
    pred_misc_iobe = os.path.join(datagen_path, 'pred_misc_iobe')
    gdth_misc_iobe = os.path.join(datagen_path, 'gdth_misc_iobe')
    label_clean(pred_file, pred_misc, ['O', 'I-MISC', 'B-MISC'], 'O')
    label_clean(gdth_file, gdth_misc, ['O', 'I-MISC', 'B-MISC'], 'O')
    iob_dict = {'O':'O', 'I':'I-MISC', 'B':'B-MISC'}
    label_translate(pred_misc, pred_misc_iobe, source_format=LabelFormat.IOB,
                    target_format=LabelFormat.IOBE, source_dict=iob_dict)
    label_translate(gdth_misc, gdth_misc_iobe, source_format=LabelFormat.IOB,
                    target_format=LabelFormat.IOBE, source_dict=iob_dict)
    _, _, _, tp, fp, fn = show_decoding_result(pred_misc_iobe, gdth_misc_iobe,
                                               subword=True)
    count_tp += tp
    count_fp += fp
    count_fn += fn

    if count_tp + count_fp != 0:
        pcs = count_tp / float(count_tp + count_fp)
    else:
        pcs = None
    if count_tp + count_fn != 0:
        recall = count_tp / float(count_tp + count_fn)
    else:
        recall = None
    try:
        f1 = 2 * pcs * recall / float(pcs + recall)
    except:
        f1 = None
    print('precision: %s, recall: %s, f1: %s' %(pcs, recall, f1))

def performance_msra():
    from config.path_config_msra import datagen_path
    pred_file = os.path.join(datagen_path, 'word_test.transformer.word2ner_msra_hparams.word2ner_msra.beam4.alpha1.0.decodes')
    gdth_file = os.path.join(datagen_path, 'ner_test')
    text_file = os.path.join(datagen_path, 'word_test')
    count_tp = 0
    count_fp = 0
    count_fn = 0

    pred_per = os.path.join(datagen_path, 'pred_per')
    gdth_per = os.path.join(datagen_path, 'gdth_per')
    pred_per_iobe = os.path.join(datagen_path, 'pred_per_iobe')
    gdth_per_iobe = os.path.join(datagen_path, 'gdth_per_iobe')
    label_clean(pred_file, pred_per, ['O', 'I-PER', 'B-PER', 'E-PER'], 'O')
    label_clean(gdth_file, gdth_per, ['O', 'I-PER', 'B-PER', 'E-PER'], 'O')
    iobe_dict = {'O':'O', 'I':'I-PER', 'B':'B-PER', 'E':'E-PER'}
    label_translate(pred_per, pred_per_iobe, source_format=LabelFormat.IOBE,
                    target_format=LabelFormat.IOBE, source_dict=iobe_dict)
    label_translate(gdth_per, gdth_per_iobe, source_format=LabelFormat.IOBE,
                    target_format=LabelFormat.IOBE, source_dict=iobe_dict)
    _, _, _, tp, fp, fn = show_decoding_result(pred_per_iobe, gdth_per_iobe,
                                               subword=True)
    count_tp += tp
    count_fp += fp
    count_fn += fn

    pred_loc= os.path.join(datagen_path, 'pred_loc')
    gdth_loc = os.path.join(datagen_path, 'gdth_loc')
    pred_loc_iobe = os.path.join(datagen_path, 'pred_loc_iobe')
    gdth_loc_iobe = os.path.join(datagen_path, 'gdth_loc_iobe')
    label_clean(pred_file, pred_loc, ['O', 'I-LOC', 'B-LOC', 'E-LOC'], 'O')
    label_clean(gdth_file, gdth_loc, ['O', 'I-LOC', 'B-LOC', 'E-LOC'], 'O')
    iobe_dict = {'O':'O', 'I':'I-LOC', 'B':'B-LOC', 'E':'E-LOC'}
    label_translate(pred_loc, pred_loc_iobe, source_format=LabelFormat.IOBE,
                    target_format=LabelFormat.IOBE, source_dict=iobe_dict)
    label_translate(gdth_loc, gdth_loc_iobe, source_format=LabelFormat.IOBE,
                    target_format=LabelFormat.IOBE, source_dict=iobe_dict)
    _, _, _, tp, fp, fn = show_decoding_result(pred_loc_iobe, gdth_loc_iobe,
                                               subword=True)
    count_tp += tp
    count_fp += fp
    count_fn += fn

    pred_org = os.path.join(datagen_path, 'pred_org')
    gdth_org = os.path.join(datagen_path, 'gdth_org')
    pred_org_iobe = os.path.join(datagen_path, 'pred_org_iobe')
    gdth_org_iobe = os.path.join(datagen_path, 'gdth_org_iobe')
    label_clean(pred_file, pred_org, ['O', 'I-ORG', 'B-ORG', 'E-ORG'], 'O')
    label_clean(gdth_file, gdth_org, ['O', 'I-ORG', 'B-ORG', 'E-ORG'], 'O')
    iobe_dict = {'O':'O', 'I':'I-ORG', 'B':'B-ORG', 'E':'E-ORG'}
    label_translate(pred_org, pred_org_iobe, source_format=LabelFormat.IOBE,
                    target_format=LabelFormat.IOBE, source_dict=iobe_dict)
    label_translate(gdth_org, gdth_org_iobe, source_format=LabelFormat.IOBE,
                    target_format=LabelFormat.IOBE, source_dict=iobe_dict)
    _, _, _, tp, fp, fn = show_decoding_result(pred_org_iobe, gdth_org_iobe,
                                               subword=True)
    count_tp += tp
    count_fp += fp
    count_fn += fn

    if count_tp + count_fp != 0:
        pcs = count_tp / float(count_tp + count_fp)
    else:
        pcs = None
    if count_tp + count_fn != 0:
        recall = count_tp / float(count_tp + count_fn)
    else:
        recall = None
    try:
        f1 = 2 * pcs * recall / float(pcs + recall)
    except:
        f1 = None
    print('precision: %s, recall: %s, f1: %s' %(pcs, recall, f1))


if __name__ == "__main__":
    #align()
    #performance_conll()
    performance()
    #performance_msra()
    pass
