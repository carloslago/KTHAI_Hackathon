from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import pandas as pd
from keras.models import load_model
import random
import os

latent_dim = 256  # Latent dimensionality of the encoding space.
firs_char = ""
max_length = 50

data = pd.read_csv(os.path.join('bazinga/static/train-balanced-sarcasm.csv'))
data.dropna(subset=['comment'], inplace=True)
data.dropna(subset=['parent_comment'], inplace=True)
data = data[data['label'] == 1]
mask = (data['comment'].str.len() <= max_length) & (data['parent_comment'].str.len() <= max_length)
data = data.loc[mask]

num_samples = len(data)  # Number of samples to train on.

parent_comments = data['parent_comment']
comments = data['comment']
full_comment = parent_comments + ' ' + comments
merged = ' '.join(full_comment.values)
raw_text = merged
chars = sorted(list(set(raw_text)))

input_texts = sorted(list(parent_comments))
target_texts = sorted(list(comments))
num_encoder_tokens = len(chars)
num_decoder_tokens = len(chars)
max_encoder_seq_length = max_length
max_decoder_seq_length = max_length

token_index = dict([(char, i) for i, char in enumerate(chars)])

encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text[:max_encoder_seq_length]):
        encoder_input_data[i, t, token_index[char]] = 1.
    encoder_input_data[i, t + 1:, token_index[' ']] = 1.
    for t, char in enumerate(target_text[:max_decoder_seq_length]):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, token_index[char]] = 1.
    decoder_input_data[i, t + 1:, token_index[' ']] = 1.
    decoder_target_data[i, t:, token_index[' ']] = 1.


# Run trainingos
model = load_model(os.path.join('bazinga/static/PLEASEWORK3.h5'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])

encoder_inputs = model.input[0]   # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output   # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]   # input_2
decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')
decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
# decoder_batch = model.layers[5]
# decoder_outputs = decoder_batch(decoder_outputs)
# decoder_dense2 = model.layers[5]
# decoder_outputs = decoder_dense2(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)


# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in token_index.items())


# Decodes an input sequence.  Future work should support beam search.
def decode_sequence(input_seq):
    global firs_char
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.

    to_choose = list(range(31, 57)) + list((range(63, 89)))
    random_start = random.choice(to_choose)
    firs_char = chars[random_start]
    target_seq[0, 0, random_start] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char
        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

def predict_input(text):
    input_seq = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    for t, char in enumerate(text[:max_encoder_seq_length]):
        input_seq[0, t, token_index[char]] = 1.
    choices = []
    for i in range(3):
        decoded_sentence = decode_sequence(input_seq)
        choices.append(firs_char+decoded_sentence)
    return choices
