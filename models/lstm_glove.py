"""Naive LSTM model."""
import keras.layers as L
import keras.backend as K
from keras.models import Model
import numpy as np
from word_dict_gen import WORD_INDEX, CONTEXT_TEXTS
import os

# pylint: disable=line-too-long

def build_model(char_size=27, dim=64, training=True, **kwargs):
  """Build the model."""
  # Inputs
  # Context: (rules, preds, chars,)
  context = L.Input(shape=(None, None, None,), name='context', dtype='int32')
  query = L.Input(shape=(None,), name='query', dtype='int32')

  var_flat = L.Lambda(lambda x: K.reshape(x, K.stack([-1, K.prod(K.shape(x)[1:])])), name='var_flat')
  flat_ctx = var_flat(context)

  print('Found %s texts.' % len(CONTEXT_TEXTS))
  word_index = WORD_INDEX
  print('Found %s unique tokens.' % len(word_index))

  embeddings_index = {}
  GLOVE_DIR = os.path.abspath('.') + "/data/glove"
  f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), 'r', encoding='utf-8')
  for line in f:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs
  f.close()

  print('Found %s word vectors.' % len(embeddings_index))

  EMBEDDING_DIM = 100

  embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
  for word, i in word_index.items():
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
          # words not found in embedding index will be all-zeros.
          embedding_matrix[i] = embedding_vector

  # Onehot embedding
  # onehot = L.Embedding(char_size, char_size,
  #                      embeddings_initializer='identity',
  #                      trainable=False,
  #                      mask_zero=True,
  #                      name='onehot')
  embedding_layer = L.Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                trainable=False)
  embedded_ctx = embedding_layer(flat_ctx) # (?, rules*preds*chars, char_size)
  embedded_q = embedding_layer(query) # (?, chars, char_size)

  # Read query
  _, *states = L.LSTM(dim, return_state=True, name='query_lstm')(embedded_q)
  # Read context
  out, *states = L.LSTM(dim, return_state=True, name='ctx_lstm')(embedded_ctx, initial_state=states)

  # Prediction
  out = L.concatenate([out]+states, name='final_states')
  out = L.Dense(1, activation='sigmoid', name='out')(out)

  model = Model([context, query], out)
  if training:
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
  return model
