from keras.models import Model
from keras.layers import Input, LSTM, Dense, BatchNormalization
import numpy as np
import pandas as pd
from keras.models import load_model

batch_size = 256  # Batch size for training.
epochs = 35  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.

max_length = 50

data = pd.read_csv('data/train-balanced-sarcasm.csv')
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
# input_token_index = dict(
#     [(char, i) for i, char in enumerate(char_parent)])
# target_token_index = dict(
#     [(char, i) for i, char in enumerate(char_comment)])

encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    try:
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, token_index[char]] = 1.
        encoder_input_data[i, t + 1:, token_index[' ']] = 1.
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, token_index[char]] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, token_index[char]] = 1.
        decoder_input_data[i, t + 1:, token_index[' ']] = 1.
        decoder_target_data[i, t:, token_index[' ']] = 1.
    except:
        pass


# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,initial_state=encoder_states)
# decoder_dense1 = Dense(256, activation='relu')
# decoder_outputs = decoder_dense1(decoder_outputs)
# decoder_batch = BatchNormalization()
# decoder_outputs = decoder_batch(decoder_outputs)
decoder_dense2 = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense2(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
print(model.summary())
# Run training
# model = load_model('s2s_2.h5')
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2, verbose=2)
# Save model
model.save('PLEASEWORK3.h5')