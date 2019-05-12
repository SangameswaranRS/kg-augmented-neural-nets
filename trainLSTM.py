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

print('[INFO] Preprocessing dataset')
# Obtaining Train data
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_labels = newsgroups_train.target
newsgroups_data = newsgroups_train.data
# Create lookup dictionary retaining top VOCABULARY_SIZE words
tokenizer.fit_on_texts(newsgroups_data)
sequences = tokenizer.texts_to_sequences(newsgroups_data)
# transform labels to one hot encoded for feeding into the neural network
newsgroups_labels_encoded = to_categorical(newsgroups_labels)
print('[INFO] Train data Vectorized')

#Pad the sequences to MAX_SEQUENCE_LENGTH Truncate sequences having length > MAX_SEQUENCE_LENGTH
padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('[INFO] Sequences Padded')

# Build Model
model = Sequential()
model.add(Embedding(VOCABULARY_SIZE, OUTPUT_EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(len(newsgroups_train.target_names), activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
print('[INFO] Training Neural Network')
history = model.fit(padded_sequences, newsgroups_labels_encoded, epochs=10)
print('[INFO] Plotting History using pyplot')
plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
print('[INFO] Serializing and writing model, History to file')
clf_file = open('classifier', 'wb')
history_file = open('history', 'wb')
pickle.dump(model,clf_file)
pickle.dump(history, history_file)
print('[INFO] Write complete')