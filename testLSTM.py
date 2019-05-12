# Import Keras Libraries
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
#Import NLTK libraries
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

print('[INFO] Necessary libraries imported')

#Config
VOCABULARY_SIZE = 5000
MAX_SEQUENCE_LENGTH = 1000
OUTPUT_EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=VOCABULARY_SIZE)

print('[INFO] Preprocessing Testing data')
# Obtaining Testing data
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_train_data = newsgroups_train.data
newsgroups_train_labels = newsgroups_train.target
newsgroups_test = fetch_20newsgroups(subset='test')
newsgroups_labels = newsgroups_test.target
newsgroups_data = newsgroups_test.data
# Create lookup dictionary retaining top VOCABULARY_SIZE words
tokenizer.fit_on_texts(newsgroups_train_data)
sequences = tokenizer.texts_to_sequences(newsgroups_data)
# transform labels to one hot encoded for feeding into the neural network
newsgroups_labels_encoded = to_categorical(newsgroups_labels)
print('[INFO] Test data Vectorized')

#Pad the sequences to MAX_SEQUENCE_LENGTH Truncate sequences having length > MAX_SEQUENCE_LENGTH
padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('[INFO] Sequences Padded Loading Model')

model_file = open('classifier', 'rb')
model = pickle.load(model_file)

print('[INFO] Model loaded ')
print(model.summary())

eval_params = model.evaluate(padded_sequences, newsgroups_labels_encoded)
print('Loss')
print(eval_params[0])
print(' Testing Accuracy: '+ str(eval_params[1]))