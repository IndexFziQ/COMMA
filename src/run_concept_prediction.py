#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@File   : run_concept_prediction.py
@Author : Yuqiang Xie
@Date   : 2021/1/24
@E-Mail : indexfziq@gmail.com
"""

import csv
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

concept2motive = './data/motive_imp.tsv'
concept2emotion = './data/emotion_imp.tsv'
test_file = './data/test.tsv'

pred_b2m = './data/result_pred_b2m.tsv'
pred_e2m = './data/result_pred_e2m.tsv'
pred_m2e = './data/result_pred_m2e.tsv'

def accuracy_f1(pred, labels):

    p, r, f1, _ = precision_recall_fscore_support(labels, pred, 1, pos_label=1, average='weighted')

    return p, r, f1


with open (concept2motive, 'r', encoding='utf-8') as cm, \
        open (concept2emotion, 'r', encoding='utf-8') as ce, \
        open (test_file, 'r', encoding='utf-8') as tf, \
        open (pred_b2m, 'w', encoding='utf-8') as bm_writer, \
        open (pred_e2m, 'w', encoding='utf-8') as em_writer, \
        open (pred_m2e, 'w', encoding='utf-8') as me_writer:

    bm_writer.write ('storyid_linenum\tchar\tmotive\temotion\tbehavior\tconcepts\tprob\n')
    em_writer.write ('storyid_linenum\tchar\tmotive\temotion\tbehavior\tconcepts\tprob\n')
    me_writer.write ('storyid_linenum\tchar\tmotive\temotion\tbehavior\tconcepts\tprob\n')
    cm_reader = csv.reader (cm, delimiter='\t')
    next(cm_reader)

    cm_lines = []
    cm_dict = {}
    for cm_line in cm_reader:
        cm_lines.append(cm_line)
        prob = [float (x) for x in cm_line[1:]]
        cm_dict[cm_line[0]]=prob

    ce_reader = csv.reader (ce, delimiter='\t')
    next (ce_reader)

    ce_lines = []
    ce_dict = {}
    for ce_line in ce_reader:
        ce_lines.append (ce_line)
        prob = [float (x) for x in ce_line[1:]]
        ce_dict[ce_line[0]]=prob

    test_reader = csv.reader (tf, delimiter='\t')
    next (test_reader)
    concepts_pred = []
    needs_labels = []
    emotions_labels = []
    needs_pred = []
    emotions_pred = []
    for test_line in test_reader:
        print(test_line)
        vertor_mot = []
        vertor_emo = []

        needs = test_line[6]
        needs_labels.append(needs)
        emotions = test_line[8]
        emotions_labels.append(emotions)

        concepts = test_line[5].split(',')
        for concept in concepts:
            if concept in cm_dict.keys():
                vertor_mot.append(cm_dict[concept])
            if concept in ce_dict.keys():
                vertor_emo.append(ce_dict[concept])

        pred_mot = np.mean(vertor_mot,axis=0)
        pred_mot_l = np.argmax(pred_mot)+1
        needs_pred.append(str(pred_mot_l))
        bm_writer.write(test_line[0]+'\t'+test_line[1]+'\t'+needs+'\t'+emotions+'\t'+test_line[4]
                        +'\t'+test_line[5]+'\t'+str(pred_mot)+'\n')
        em_writer.write (
            test_line[0] + '\t' + test_line[1] + '\t' + needs + '\t' + emotions + '\t' + test_line[4] + '\t' +
            test_line[5] + '\t' + str(pred_mot) + '\n')
        pred_emo = np.mean(vertor_emo,axis=0)
        pred_emo_l = np.argmax (pred_emo)+1
        emotions_pred.append(str(pred_emo_l))
        me_writer.write (
            test_line[0] + '\t' + test_line[1] + '\t' + needs + '\t' + emotions + '\t' + test_line[3] + '\t' +
            test_line[5]+'\t'+str(pred_mot) + '\t' + str(pred_emo) + '\n')

        print(pred_mot)
        print(pred_emo)
        print(pred_mot_l)
        print(pred_emo_l)

    p, r, f1 = accuracy_f1(needs_pred, needs_labels)
    print(p)
    print(r)
    print(f1)

    p_, r_, f1_ = accuracy_f1(emotions_pred, emotions_labels)
    print(p_)
    print(r_)
    print(f1_)







