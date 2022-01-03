import os
import json_lines
import re
from keras.preprocessing.text import Tokenizer

# Contextual embeddeding of symbols
texts = []  # list of text samples
id_list = []
question_list = []
label_list = []
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
#TEXT_DATA_DIR = os.path.abspath('.') + "/data/pararule"
#TEXT_DATA_DIR = "/data/qbao775/deeplogic-master/data/pararule"
TEXT_DATA_DIR = "./data/ConceptRules"

print(TEXT_DATA_DIR)
# TEXT_DATA_DIR = "D:\\AllenAI\\20_newsgroup"
Str = '.jsonl'
CONTEXT_TEXTS = []
#test_str = 'test'
meta_str = 'meta'

for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            fpath = os.path.join(path, fname)
            if Str in fpath:
                #if test_str not in fpath:
                if meta_str not in fpath:
                    with open(fpath) as f:
                        for l in json_lines.reader(f):
                            #if l["id"] not in id_list:
                            id_list.append(l["id"])
                            questions = l["questions"]
                            context = l["context"].replace("\n", " ").replace(".", "")
                            context = context.replace(",", "")
                            context = context.replace("!", "")
                            context = context.replace("\\", "")
                            context = re.sub(r'\s+', ' ', context)
                            CONTEXT_TEXTS.append(context)
                            # for i in range(len(questions)):
                            #     text = questions[i]["text"]
                            #     label = questions[i]["label"]
                            #     if label == True:
                            #         t = 1
                            #     else:
                            #         t = 0
                            #     q = re.sub(r'\s+', ' ', text)
                            #     q = q.replace(',', '')
                            #     texts.append(context)
                            #     question_list.append(q)
                            #     label_list.append(int(t))
                    f.close()
        # labels.append(label_id)

print('Found %s texts.' % len(CONTEXT_TEXTS))

MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1000
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS,lower=True,filters="")
tokenizer.fit_on_texts(CONTEXT_TEXTS)
# sequences = tokenizer.texts_to_sequences(texts)

WORD_INDEX = tokenizer.word_index
#WORD_INDEX['v'] = 10344
print('Found %s unique tokens.' % len(WORD_INDEX))
#print('Found %s unique tokens.' % len(WORD_INDEX))