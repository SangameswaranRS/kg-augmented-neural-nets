import keras
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers import Conv3D, MaxPooling3D
from keras.layers import Conv2D, MaxPooling2D
# from keras.layers import Flatten


def custom_LSTM_neural_network():
    # Configuration for Neural network
    VOCABULARY_SIZE = 5000
    MAX_SEQUENCE_LENGTH = 1000
    OUTPUT_EMBEDDING_DIM = 100
    
    # Create a sequential model
    custom_LSTM_model = Sequential()
    custom_LSTM_model.add(Embedding(VOCABULARY_SIZE, OUTPUT_EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    custom_LSTM_model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    custom_LSTM_model.add(Dense(20, activation='sigmoid'))
    custom_LSTM_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #Return the compiled neural network model
    return custom_LSTM_model

def custom_3D_CNN():
    custom_3D_CNN_model = Sequential()
    custom_3D_CNN_model.add(Conv3D(filters=100, kernel_size=(5,5,5), input_shape=(299,2,10,1),padding='same'))
    custom_3D_CNN_model.add(MaxPooling3D(pool_size=(2,2,2)))
    custom_3D_CNN_model.add(Dropout(0.2))
    custom_3D_CNN_model.add(Flatten())
    custom_3D_CNN_model.add(Dense(100, activation='relu'))
    custom_3D_CNN_model.add(Dense(20, activation='sigmoid'))
    custom_3D_CNN_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), metrics=['accuracy'])
    return custom_3D_CNN_model

def custom_1D_CNN():
    # Configuration for Neural network
    VOCABULARY_SIZE = 5000
    MAX_SEQUENCE_LENGTH = 300
    OUTPUT_EMBEDDING_DIM = 100
    # Build the model
    custom_1D_CNN_model = Sequential()
    custom_1D_CNN_model.add(Embedding(VOCABULARY_SIZE, OUTPUT_EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    custom_1D_CNN_model.add(Conv1D(filters=100, kernel_size=5, strides=1, padding='same', activation='relu'))
    custom_1D_CNN_model.add(MaxPooling1D(pool_size=2, strides= None))
    custom_1D_CNN_model.add(Dropout(0.2))
    custom_1D_CNN_model.add(Conv1D(filters=100,kernel_size=5))
    custom_1D_CNN_model.add(MaxPooling1D(pool_size=2, strides=1))
    custom_1D_CNN_model.add(Dropout(0.3))
    custom_1D_CNN_model.add(Flatten())
    custom_1D_CNN_model.add(Dense(100, activation='relu'))

    #custom_1D_CNN_model.add(Conv1D(50, 5, strides=1))
    #custom_1D_CNN_model.add(MaxPooling1D(pool_size=3, strides=None))
    custom_1D_CNN_model.add(Dense(20,activation='sigmoid'))
    custom_1D_CNN_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Return the model
    return custom_1D_CNN_model

def custom_C_LSTM_model():
    # Configuration for Neural network
    VOCABULARY_SIZE = 5000
    MAX_SEQUENCE_LENGTH = 1000
    OUTPUT_EMBEDDING_DIM = 32
    
    # Create a sequential model
    custom_cLSTM_model = Sequential()
    custom_cLSTM_model.add(Embedding(VOCABULARY_SIZE, OUTPUT_EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    custom_cLSTM_model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'))
    custom_cLSTM_model.add(MaxPooling1D(pool_size=2))
    custom_cLSTM_model.add(LSTM(100))
    custom_cLSTM_model.add(Dense(20, activation='sigmoid'))
    custom_cLSTM_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #Return the compiled neural network model
    return custom_cLSTM_model
