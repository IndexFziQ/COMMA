import numpy as np

in_file = '/home/pengwei.pw/third/examples/seq2seq/test_data/wmt_en_ro/train.source'
with open(in_file, 'r', encoding='utf-8') as in_file:
    for text in in_file.readlines():
        print(text)