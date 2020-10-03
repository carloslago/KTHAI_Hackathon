import os
import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint

data = pd.read_csv('data/train-balanced-sarcasm.csv')
data.dropna(subset=['comment'], inplace=True)
data.dropna(subset=['parent_comment'], inplace=True)
data = data[data['label'] == 1]
data = data[data['subreddit'].isin(['funny', 'politics'])]


parent_comments = data['parent_comment']
comments = data['comment']
# full_comment = parent_comments + ' ' + comments
full_comment = comments
merged = ' '.join(full_comment.values)
raw_text = merged
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
n_chars = len(raw_text)
n_vocab = len(chars)

seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
X = np.reshape(dataX, (n_patterns, seq_length, 1))
X = X / float(n_vocab)
y = np_utils.to_categorical(dataY)
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
filepath="test_model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.fit(X, y, epochs=10, batch_size=2048, verbose=2, callbacks=callbacks_list)

# int_to_char = dict((i, c) for i, c in enumerate(chars))


