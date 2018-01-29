# coding:utf-8
from environment import *


def compute_performance(label, pred_label):
    '''1：begin；2：medium；3：end；0：others'''

    def gen_entities(l):
        """extract entities in a label sequence as a dictionary
           key: span
           value: label sub-sequence"""
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
        return entities

    true_entities = gen_entities(label)
    pred_entities = gen_entities(pred_label)

    num_true = len(true_entities)
    num_extraction = len(pred_entities)

    num_true_positive = 0
    for pred_entity in pred_entities.keys():
        try:
            if true_entities[pred_entity] == pred_entities[pred_entity]:
                num_true_positive += 1
        except:
            pass

    acc = len(filter(lambda x: x[0] == x[1], zip(label, pred_label))) / float(len(label))
    if num_extraction != 0:
        pcs = num_true_positive/float(num_extraction)
    else:
        print 'ATTENTION: no legal entity predicted'
        pcs = 0
    if num_true != 0:
        recall = num_true_positive/float(num_true)
    else:
        print 'ATTENTION: no legal entity'
        recall = 0

    if pcs + recall != 0:
        f1 = 2 * pcs * recall / (pcs + recall)
    else:
        f1 = 0
        print 'ATTENTION: no legal entity in both'

    return acc, pcs, recall, f1


def compute_batch_performance(label_matrix, mask_matrix, pred_label_matrix):
    '''1：begin；2：medium；3：end；0：others'''
    whole_label = []
    whole_pred_label = []
    for i in range(label_matrix.shape[0]):
        mask = list(mask_matrix[i])
        label = list(label_matrix[i] * mask)
        pred_label = list(pred_label_matrix[i] * mask)

        label, _ = zip(*filter(lambda x: x[1] == 1, zip(label, mask)))
        pred_label, _ = zip(*filter(lambda x: x[1] == 1, zip(pred_label, mask)))
        whole_label.extend(label)
        whole_pred_label.extend(pred_label)
    acc, pcs, recall, f1 = compute_performance(whole_label, whole_pred_label)
    return acc, pcs, recall, f1


if __name__ == '__main__':
    # label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # pred_label = [0, 0, 0, 0, 0, 1, 2, 3, 0, 0]
    label_matrix = np.array([[1, 2, 3, 0, 0], [0, 1, 2, 3, 0]])
    mask_matrix = np.array([[1, 1, 1, 1, 0], [1, 0, 0, 0, 0]])
    pred_label_matrix = np.array([[1, 2, 3, 0, 0], [0, 0, 0, 0, 0]])
    print compute_batch_performance(label_matrix, mask_matrix, pred_label_matrix)
