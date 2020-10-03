from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import pandas as pd
from keras.models import load_model

latent_dim = 256  # Latent dimensionality of the encoding space.

data = pd.read_csv('data/train-balanced-sarcasm.csv')
data.dropna(subset=['comment'], inplace=True)
data.dropna(subset=['parent_comment'], inplace=True)
data = data[data['label'] == 1]
mask = (data['comment'].str.len() <= 50) & (data['parent_comment'].str.len() <= 50)
data = data.loc[mask]

num_samples = len(data)  # Number of samples to train on.

parent_comments = data['parent_comment']
comments = data['comment']
full_comment = parent_comments + ' ' + comments
merged = ' '.join(full_comment.values)
raw_text = merged
chars = sorted(list(set(raw_text)))

char_parent = sorted(list(set(list(parent_comments))))
char_comment = sorted(list(set(comments)))

input_texts = list(parent_comments)
target_texts = list(comments)
num_encoder_tokens = len(chars)
num_decoder_tokens = len(chars)
max_encoder_seq_length = 50
max_decoder_seq_length = 50

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


# Run training
model = load_model('PLEASEWORK2.h5')
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
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, token_index[' ']] = 1.

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


for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)