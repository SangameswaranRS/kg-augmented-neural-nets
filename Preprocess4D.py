from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.datasets import fetch_20newsgroups
import pickle
import numpy as np

VOCABULARY_SIZE = 5000
MAX_SEQUENCE_LENGTH = 300

print('[INFO] Fetching Training Data')
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_train_data = newsgroups_train.data
sequence_tokenizer = Tokenizer(num_words = VOCABULARY_SIZE)
sequence_tokenizer.fit_on_texts(newsgroups_train_data)
print('[INFO] Vectorizing data')
newsgroups_train_data_sequence_unpadded = sequence_tokenizer.texts_to_sequences(newsgroups_train_data)
print('[INFO] Padding Sequences')
newsgroups_train_data_sequence_padded = pad_sequences(newsgroups_train_data_sequence_unpadded, maxlen=MAX_SEQUENCE_LENGTH)
print('[INFO] Sequences Padded')
word2vec_model_file = open('word2vec_model', 'rb')
word2vec_model = pickle.load(word2vec_model_file)
print('[INFO] Pre-Trained word2vec model loaded')
print('[INFO] Changing padded sequences back to words')
# sequence_tokenizer.index_word.get(1)
texts = []
i = 0
for seq in newsgroups_train_data_sequence_padded:
    i=i+1
    curr_word_list = []
    for num in seq:
        if num == 0:
            curr_word_list.append('<PAD>')
        else:
            curr_word_list.append(sequence_tokenizer.index_word.get(num))
        
    texts.append(curr_word_list)
print('[INFO] Texts padded and normalized')

def retreive_word2vec(word):
    zero_vector = np.zeros((10,), dtype='float32')
    try:
        vec = word2vec_model.wv[word]
        return vec
    except Exception as E:
        return zero_vector
def induividualTextArrayTo3D(text):
    d3_record_vec = []
    for i in range(0,len(text)-1):
        bigram_2d_vec = [retreive_word2vec(text[i]),retreive_word2vec(text[i+1])]
        d3_record_vec.append(bigram_2d_vec)
    d3_numpy_array = np.array(d3_record_vec)
    print(d3_numpy_array.shape)
    return d3_numpy_array
        
def construct4D_text():
    d4_record_vec = []
    for text in texts:
        d3_numpy = induividualTextArrayTo3D(text)
        d4_record_vec.append(d3_numpy)
    d4_numpy = np.array(d4_record_vec)
    print(d4_numpy.shape)
    return d4_numpy
print('[INFO] Constructing 4D input')
d4_processed = construct4D_text()
print('[INFO] d4 Processed')
print('[INFO] Processing d5')
d5_input = d4_processed.reshape(11314,299,2,10,1)
d5_input_file = open('d5_input','wb')
pickle.dump(d5_input, d5_input_file)
print('[INFO] d5 Processed and saved')


