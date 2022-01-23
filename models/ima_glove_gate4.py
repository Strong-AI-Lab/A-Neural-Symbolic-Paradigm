"""Iterative memory attention model."""
import numpy as np
import keras.backend as K
import keras.layers as L
from keras.models import Model
import tensorflow as tf
import random
import random as python_random
import os
from word_dict_gen import WORD_INDEX, CONTEXT_TEXTS

from .zerogru import ZeroGRU, NestedTimeDist

# pylint: disable=line-too-long
def build_model(char_size=27, dim=64, iterations=4, training=True, ilp=False, pca=False):
  """Build the model."""
  # Inputs
  # Context: (rules, preds, chars,)
  context = L.Input(shape=(None, None, None,), name='context', dtype='int32')
  query = L.Input(shape=(None,), name='query', dtype='int32')

  if ilp:
    context, query, templates = ilp

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

  embedding_layer = L.Embedding(len(word_index) + 1,
                              EMBEDDING_DIM,
                              weights=[embedding_matrix],
                              trainable=False)

  # Contextual embeddeding of symbols
  # onehot_weights = np.eye(char_size)
  # onehot_weights[0, 0] = 0 # Clear zero index
  # onehot = L.Embedding(char_size, char_size,
  #                      trainable=False,
  #                      weights=[onehot_weights],
  #                      name='onehot')
  # embedded_ctx = onehot(context) # (?, rules, preds, chars, char_size)
  # embedded_q = onehot(query) # (?, chars, char_size)
  embedded_ctx = embedding_layer(context) # (?, rules, preds, chars, char_size)
  embedded_q = embedding_layer(query) # (?, chars, char_size)
  K.print_tensor(embedded_q)
  if ilp:
    # Combine the templates with the context, (?, rules+temps, preds, chars, char_size)
    embedded_ctx = L.Lambda(lambda xs: K.concatenate(xs, axis=1), name='template_concat')([templates, embedded_ctx])
    # embedded_ctx = L.concatenate([templates, embedded_ctx], axis=1)

  embed_pred = ZeroGRU(dim, go_backwards=True, name='embed_pred')
  embedded_predq = embed_pred(embedded_q) # (?, dim)
  # For every rule, for every predicate, embed the predicate
  embedded_ctx_preds = NestedTimeDist(NestedTimeDist(embed_pred, name='nest1'), name='nest2')(embedded_ctx)
  # (?, rules, preds, dim)

  # embed_rule = ZeroGRU(dim, go_backwards=True, name='embed_rule')
  # embedded_rules = NestedTimeDist(embed_rule, name='d_embed_rule')(embedded_ctx_preds)
  get_heads = L.Lambda(lambda x: x[:, :, 0, :], name='rule_heads')
  embedded_rules = get_heads(embedded_ctx_preds)
  # (?, rules, dim)

  # Reused layers over iterations
  rule_to_att = L.TimeDistributed(L.Dense(dim//2, name='rule_to_att'), name='d_rule_to_att')
  state_to_att = L.Dense(dim//2, name='state_to_att')
  repeat_toctx = L.RepeatVector(K.shape(context)[1], name='repeat_to_ctx')
  att_dense = L.TimeDistributed(L.Dense(1), name='att_dense')
  squeeze2 = L.Lambda(lambda x: K.squeeze(x, 2), name='sequeeze2')

  unifier = NestedTimeDist(ZeroGRU(dim, name='unifier'), name='dist_unifier')
  # dot11 = L.Dot((1, 1))
  gating = L.Dense(1, activation='sigmoid', name='gating')
  gate2 = L.Lambda(lambda xyg: xyg[2]*xyg[0] + (1-xyg[2])*xyg[1], name='gate')

  # Reasoning iterations
  state = L.Dense(dim, activation='tanh', name='init_state')(embedded_predq)
  ctx_rules = rule_to_att(embedded_rules)
  outs = list()
  for _ in range(iterations):
    # Compute attention between rule and query state
    att_state = state_to_att(state)  # (?, ATT_LATENT_DIM)
    att_state = repeat_toctx(att_state)  # (?, rules, ATT_LATENT_DIM)
    sim_vec = L.multiply([ctx_rules, att_state])
    sim_vec = att_dense(sim_vec)  # (?, rules, 1)
    sim_vec = squeeze2(sim_vec)  # (?, rules)
    sim_vec = L.Softmax(axis=1)(sim_vec)
    outs.append(sim_vec)

    # Unify every rule and weighted sum based on attention
    new_states = unifier(embedded_ctx_preds, initial_state=[state])
    # (?, rules, dim)
    new_state = L.dot([sim_vec, new_states], (1, 1))
    s_m_ns = L.multiply([state, new_state])
    s_s_ns = L.subtract([state, new_state])
    gate = L.concatenate([state, new_state, s_m_ns, s_s_ns])
    gate = gating(gate)
    outs.append(gate)
    new_state = gate2([state, new_state, gate])
    state = new_state

    # Apply gating
    # gate = gating(state)
    # outs.append(gate)
    # state = gate2([state, new_state, gate])

  # Predication
  out = L.Dense(1, activation='sigmoid', name='out')(state)
  if ilp:
    return outs, out
  elif pca:
    model = Model([context, query], [embedded_rules])
  elif training:
    model = Model([context, query], [out])
    # optimizer = tf.keras.optimizers.Adam(lr=0.01)
    model.compile(loss='binary_crossentropy',
                  optimizer="adam",
                  metrics=['acc'])
  else:
    model = Model([context, query], outs + [out])
  return model
