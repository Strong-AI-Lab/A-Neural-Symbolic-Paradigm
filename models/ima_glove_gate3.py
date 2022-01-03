"""Iterative memory attention model with Glove as pre-training embedding."""
import numpy as np
import keras.backend as K
import keras.layers as L
from keras.models import Model
from keras.optimizers import adam
import os
import re
import json_lines
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import *
from keras.layers import Embedding
from word_dict_gen import WORD_INDEX, CONTEXT_TEXTS

from .zerogru import ZeroGRU, NestedTimeDist

# pylint: disable=line-too-long

def build_model(char_size=27, dim=64, iterations=4, training=True, ilp=False, pca=False):
  """Build the model."""
  # Inputs
  # Context: (rules, preds, chars,)
  # context = L.Input(shape=(None, None, None,), name='context', dtype='int32')
  # query = L.Input(shape=(None,), name='query', dtype='int32')

  if ilp:
    context, query, templates = ilp

  # Contextual embeddeding of symbols
  # texts = []  # list of text samples
  # id_list = []
  # question_list = []
  # label_list = []
  # labels_index = {}  # dictionary mapping label name to numeric id
  # labels = []  # list of label ids
  # TEXT_DATA_DIR = os.path.abspath('.') + "/data/pararule"
  # # TEXT_DATA_DIR = "D:\\AllenAI\\20_newsgroup"
  # Str = '.jsonl'
  # CONTEXT_TEXTS = []
  # test_str = 'test'
  # meta_str = 'meta'

  # for name in sorted(os.listdir(TEXT_DATA_DIR)):
  #   path = os.path.join(TEXT_DATA_DIR, name)
  #   if os.path.isdir(path):
  #     label_id = len(labels_index)
  #     labels_index[name] = label_id
  #     for fname in sorted(os.listdir(path)):
  #       fpath = os.path.join(path, fname)
  #       if Str in fpath:
  #         if test_str not in fpath:
  #           if meta_str not in fpath:
  #             with open(fpath) as f:
  #               for l in json_lines.reader(f):
  #                 if l["id"] not in id_list:
  #                   id_list.append(l["id"])
  #                   questions = l["questions"]
  #                   context = l["context"].replace("\n", " ")
  #                   context = re.sub(r'\s+', ' ', context)
  #                   CONTEXT_TEXTS.append(context)
  #                   for i in range(len(questions)):
  #                     text = questions[i]["text"]
  #                     label = questions[i]["label"]
  #                     if label == True:
  #                       t = 1
  #                     else:
  #                       t = 0
  #                     q = re.sub(r'\s+', ' ', text)
  #                     texts.append(context)
  #                     question_list.append(q)
  #                     label_list.append(int(t))
  #             f.close()
  #       # labels.append(label_id)

  print('Found %s texts.' % len(CONTEXT_TEXTS))

  # MAX_NB_WORDS = 20000
  # MAX_SEQUENCE_LENGTH = 1000
  # tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
  # tokenizer.fit_on_texts(texts)
  # #sequences = tokenizer.texts_to_sequences(texts)

  word_index = WORD_INDEX
  print('Found %s unique tokens.' % len(word_index))

  #data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

  # labels = to_categorical(np.asarray(labels))
  #print('Shape of data tensor:', data.shape)
  # print('Shape of label tensor:', labels.shape)

  # split the data into a training set and a validation set
  # indices = np.arange(data.shape[0])
  # np.random.shuffle(indices)
  # data = data[indices]
  # labels = labels[indices]

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

  context = L.Input(shape=(None, None, None,), name='context', dtype='int32')
  query = L.Input(shape=(None,), name='query', dtype='int32')

  embedded_ctx = embedding_layer(context) # (?, rules, preds, chars, char_size)
  embedded_q = embedding_layer(query) # (?, chars, char_size)
  #onehot_weights = np.eye(char_size)
  #onehot_weights[0, 0] = 0 # Clear zero index
  # onehot = L.Embedding(char_size, char_size,
  #                      trainable=False,
  #                      weights=[onehot_weights],
  #                      name='onehot')
  # embedded_ctx = onehot(context) # (?, rules, preds, chars, char_size)
  # embedded_q = onehot(query) # (?, chars, char_size)

  if ilp:
    # Combine the templates with the context, (?, rules+temps, preds, chars, char_size)
    embedded_ctx = L.Lambda(lambda xs: K.concatenate(xs, axis=1), name='template_concat')([templates, embedded_ctx])
    # embedded_ctx = L.concatenate([templates, embedded_ctx], axis=1)

  embed_pred = ZeroGRU(dim, go_backwards=True, name='embed_pred')
  embedded_predq = embed_pred(embedded_q) # (?, dim)
  # For every rule, for every predicate, embed the predicate
  embedded_ctx_preds = L.TimeDistributed(L.TimeDistributed(embed_pred, name='nest1'), name='nest2')(embedded_ctx)
  # (?, rules, preds, dim)

  # embed_rule = ZeroGRU(dim, go_backwards=True, name='embed_rule')
  # embedded_rules = NestedTimeDist(embed_rule, name='d_embed_rule')(embedded_ctx_preds)
  get_heads = L.Lambda(lambda x: x[:, :, 0, :], name='rule_heads')
  embedded_rules = get_heads(embedded_ctx_preds)
  # (?, rules, dim)

  # Reused layers over iterations
  repeat_toctx = L.RepeatVector(K.shape(embedded_ctx)[1], name='repeat_to_ctx')
  diff_sq = L.Lambda(lambda xy: K.square(xy[0]-xy[1]), output_shape=(None, dim), name='diff_sq')
  mult = L.Multiply()
  concat = L.Lambda(lambda xs: K.concatenate(xs, axis=2), output_shape=(None, dim*5), name='concat')
  att_densel = L.Dense(dim//2, activation='tanh', name='att_densel', activity_regularizer="l1")
  att_dense = L.Dense(1, activation='sigmoid', name='att_dense', activity_regularizer="l1")
  squeeze2 = L.Lambda(lambda x: K.squeeze(x, 2), name='sequeeze2')
  rule_mask = L.Lambda(lambda x: K.cast(K.any(K.not_equal(x, 0), axis=-1, keepdims=True), 'float32'), name='rule_mask')(embedded_rules)

  unifier = NestedTimeDist(ZeroGRU(dim, name='unifier'), name='dist_unifier')
  dot11 = L.Dot((1, 1))
  gating = L.Dense(1, activation='sigmoid', name='gating', activity_regularizer="l1")
  gate2 = L.Lambda(lambda xyg: xyg[2]*xyg[0] + (1-xyg[2])*xyg[1], name='gate')

  # Reasoning iterations
  state = embedded_predq
  repeated_q = repeat_toctx(embedded_predq)
  outs = list()
  for _ in range(iterations):
    # Compute attention between rule and query state
    ctx_state = repeat_toctx(state) # (?, rules, dim)
    s_s_c = diff_sq([ctx_state, embedded_rules])
    s_m_c = mult([embedded_rules, state]) # (?, rules, dim)
    sim_vec = concat([s_s_c, s_m_c, ctx_state, embedded_rules, repeated_q])
    sim_vec = att_densel(sim_vec) # (?, rules, dim//2)
    sim_vec = att_dense(sim_vec) # (?, rules, 1)
    sim_vec = mult([sim_vec, rule_mask])
    sim_vec = squeeze2(sim_vec) # (?, rules)
    # sim_vec = L.Softmax(axis=1)(sim_vec)
    outs.append(sim_vec)

    # Unify every rule and weighted sum based on attention
    new_states = unifier(embedded_ctx_preds, initial_state=[state])
    # (?, rules, dim)
    new_state = dot11([sim_vec, new_states])
    # Apply gating
    gate = gating(new_state)
    outs.append(gate)
    new_state = gate2([new_state, state, gate])
    state = new_state

  # Predication
  out = L.Dense(1, activation='sigmoid', name='out', activity_regularizer="l1")(state)
  if ilp:
    return outs, out
  elif pca:
    model = Model([context, query], [embedded_rules])
  elif training:
    model = Model([context, query], [out])
    # opt = adam(lr=0.00001)
    model.compile(loss='binary_crossentropy',
                  optimizer="adam",
                  metrics=['acc'])
  else:
    model = Model([context, query], outs + [out])
  return model
