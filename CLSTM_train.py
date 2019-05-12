from NeuralNetworkModels import custom_1D_CNN
from NeuralNetworkModels import custom_C_LSTM_model
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

# Configuration
VOCABULARY_SIZE = 5000
MAX_SEQUENCE_LENGTH = 1000
OUTPUT_EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=VOCABULARY_SIZE)

print('[INFO] Preprocessing data')
newsgroups_train = fetch_20newsgroups(subset='train')
training_labels = newsgroups_train.target
training_data = newsgroups_train.data
# Prepare the tokenizer
tokenizer.fit_on_texts(training_data)
newsgroups_train_sequences_unpadded = tokenizer.texts_to_sequences(training_data)
# Pad the sequences
newsgroups_train_data_padded = pad_sequences(newsgroups_train_sequences_unpadded, maxlen=MAX_SEQUENCE_LENGTH)
# Transform the labels to one hot encoded form
newsgroups_labels_encoded = to_categorical(training_labels)
print('[INFO] Sequences Vectorized and padded, Categories - One hot encoded')
# Import the model
model = custom_C_LSTM_model()
model.summary()
# train the model
print('[INFO] Training CNN')
train_history = model.fit(newsgroups_train_data_padded, newsgroups_labels_encoded, epochs=10)
print('[INFO] CNN train complete. Serializing and writing model, history')
clstmmodel = open('clstm_model', 'wb')
clstmhistory = open('clstm_history', 'wb')
pickle.dump(model,clstmmodel)
pickle.dump(train_history, clstmhistory)
print('[INFO] Models Serialized and written to file')
plt.plot(train_history.history['acc'])
plt.xlabel('epoch')
plt.ylabel('trainingAccuracy')
plt.show()