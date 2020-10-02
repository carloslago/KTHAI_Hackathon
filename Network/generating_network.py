import os
import numpy as np
import pandas as pd

sarcasm_comments = []
data = pd.read_csv('data/train-balanced-sarcasm.csv')
data.dropna(subset=['comment'], inplace=True)
data.dropna(subset=['parent_comment'], inplace=True)
data = data[data['label'] == 1]

parent_comments = data['parent_comment']
comments = data['comment']
full_comment = parent_comments + ' ' + comments
merged = ' '.join(full_comment.values)
raw_text = merged
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
n_chars = len(raw_text)
n_vocab = len(chars)
# count=0
# for i in full_comment:
#     print(i + '\n')
#     count+=1
#     if count>50: break




