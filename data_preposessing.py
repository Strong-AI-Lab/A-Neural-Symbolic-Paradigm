import os
import re
import json_lines
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.utils.np_utils import *
from keras.layers import Embedding
import keras.layers as L
from models.zerogru import ZeroGRU, NestedTimeDist

texts = []  # list of text samples
id_list = []
question_list = []
label_list = []
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
TEXT_DATA_DIR = os.path.abspath(os.path.dirname(__file__)) + "\\data\\pararule"
#TEXT_DATA_DIR = "D:\\AllenAI\\20_newsgroup"
Str='.jsonl'
context_texts = []
test_str = 'test'
meta_str = 'meta'

for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            fpath = os.path.join(path, fname)
            if Str in fpath:
                if test_str not in fpath:
                    if meta_str not in fpath:
                        with open(fpath) as f:
                            for l in json_lines.reader(f):
                                if l["id"] not in id_list:
                                    id_list.append(l["id"])
                                    questions = l["questions"]
                                    ctx = list()
                                    context = l["context"].replace("\n", " ")
                                    context = re.sub(r'\s+', ' ', context)
                                    context_texts.append(context)
                                    for i in range(len(questions)):
                                        text = questions[i]["text"]
                                        label = questions[i]["label"]
                                        if label == True:
                                            t = 1
                                        else:
                                            t = 0
                                        q = re.sub(r'\s+', ' ', text)
                                        texts.append(context)
                                        question_list.append(q)
                                        label_list.append(int(t))
                        f.close()
            #labels.append(label_id)

print('Found %s texts.' % len(context_texts))

MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1000
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

#labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
#print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
#labels = labels[indices]


embeddings_index = {}
GLOVE_DIR = os.path.abspath(os.path.dirname(__file__)) + "\\data\\glove"
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'),'r',encoding='utf-8')
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

# embedding_layer = Embedding(len(word_index) + 1,
#                             EMBEDDING_DIM,
#                             weights=[embedding_matrix],
#                             input_length=MAX_SEQUENCE_LENGTH,
#                             trainable=False)

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=False)


context = L.Input(shape=(None, None, None,), name='context', dtype='int32')
query = L.Input(shape=(None,), name='query', dtype='int32')

embedded_ctx = embedding_layer(context)  # (?, rules, preds, chars, char_size)
embedded_q = embedding_layer(query)  # (?, chars, char_size)

dim=64
#dim=MAX_SEQUENCE_LENGTH

embed_pred = ZeroGRU(dim, go_backwards=True, name='embed_pred')
embedded_predq = embed_pred(embedded_q)  # (?, dim)
# For every rule, for every predicate, embed the predicate
embedded_ctx_preds = L.TimeDistributed(L.TimeDistributed(embed_pred, name='nest1'), name='nest2')(embedded_ctx)
# (?, rules, preds, dim)