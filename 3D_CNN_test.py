import pickle
import numpy as np
from NeuralNetworkModels import custom_3D_CNN
from sklearn.datasets import fetch_20newsgroups
from keras.utils.np_utils import to_categorical

# Get the preprocessed input
print('[INFO] Loading Preprocessed test input')
d5_input = pickle.load(open('d5_input', 'rb'))
print(d5_input.shape)
newsgroups_test = fetch_20newsgroups(subset='train')
newsgroups_data = newsgroups_test.data
newsgroups_labels =newsgroups_test.target
newsgroups_labels_one_hot = to_categorical(newsgroups_labels)
print('[INFO] Inputs loaded, Targets - one hot encoded..')
print('[INFO] Loading Pre- trained model for testing..')
model_file = open('3d_cnn_model.pickle','rb')
model = pickle.load(model_file)
print(model.summary())
#Evaluate the performance on test data. Train Acc was 0.62
evaluation_params = model.evaluate(d5_input,newsgroups_labels_one_hot)
print(evaluation_params)
print('[INFO] Testing Accuracy: '+str(evaluation_params[1]))
print('[DONE] Testing done.')