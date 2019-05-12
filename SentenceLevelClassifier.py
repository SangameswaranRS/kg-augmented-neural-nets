from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
import pickle
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
print('[INFO] Necessary libraries imported')
VOCABULARY_SIZE = 5000
MAX_SEQUENCE_LENGTH = 300
OUTPUT_EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=VOCABULARY_SIZE)
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_train_data = newsgroups_train.data
newsgroups_train_labels = newsgroups_train.target
tokenizer.fit_on_texts(newsgroups_train_data)
text_to_be_classified = input('Enter Document/ Text to be classified: ')
texts_to_be_classified = []
texts_to_be_classified.append(text_to_be_classified)
sequences = tokenizer.texts_to_sequences(texts_to_be_classified)
print('[INFO] Data Vectorized')
print(sequences)
padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('[INFO] Sequence Padded Loading Model')
file = open('cnn_1d_model', 'rb')
model = pickle.load(file)
print('[INFO] Model loaded')
model.summary()
eval_params = model.predict(padded_sequences)
print(newsgroups_train.target_names[np.argmax(eval_params[0])])