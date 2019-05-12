import pickle
import numpy as np
from NeuralNetworkModels import custom_3D_CNN
from sklearn.datasets import fetch_20newsgroups
from keras.utils.np_utils import to_categorical

# Get the preprocessed input
print('[INFO] Loading Preprocessed input')
d5_input = pickle.load(open('d5_input', 'rb'))
print(d5_input.shape)
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_data = newsgroups_train.data
newsgroups_labels =newsgroups_train.target
newsgroups_labels_one_hot = to_categorical(newsgroups_labels)
print('[INFO] Inputs loaded, Targets - one hot encoded')

custom_3D_CNN_model = custom_3D_CNN()
custom_3D_CNN_model.summary()
custom_3D_CNN_model.fit(d5_input,newsgroups_labels_one_hot, epochs=25, batch_size=32)
file = open('3d_cnn_model.pickle', 'wb')
pickle.dump(custom_3D_CNN_model,file)
print('[DONE] Model serialized and written to disk')