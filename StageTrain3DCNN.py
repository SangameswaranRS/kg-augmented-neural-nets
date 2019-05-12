import keras
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
print('[INFO] Retreiving model trained in the previous stage..')
prev_3d_cnn_model_file = open('3d_cnn_model.pickle', 'rb')
prev_3d_cnn_model = pickle.load(prev_3d_cnn_model_file)
print('[INFO] Prev Stage model loaded.. Extracting weights')
weights= []
for layer in prev_3d_cnn_model.layers:
    indl_layer_wt = layer.get_weights()
    weights.append(indl_layer_wt)
for i in range(0,len(custom_3D_CNN_model.layers)):
    custom_3D_CNN_model.layers[i].set_weights(weights[i])
print('[INFO] Pretrained Weight loaded, using adam Optimizer!')
print('[INFO] Stage training')
custom_3D_CNN_model.fit(d5_input,newsgroups_labels_one_hot, epochs=25, batch_size=64, callbacks=[keras.callbacks.EarlyStopping(monitor='acc', min_delta=0.0001, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=True)])  
print('[INFO] Training Complete')
new_model_file = open('3d_cnn_modelv1.pickle','wb')
pickle.dump(custom_3D_CNN_model, new_model_file)
print('[INFO] Classifier Successfully stage trained')
