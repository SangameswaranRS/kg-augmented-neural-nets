import gensim
import pickle
from sklearn.datasets import fetch_20newsgroups
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

print('[INFO] Necessary Libraries Imported')

# Fetch the training data
training_data = fetch_20newsgroups(subset='train')
print('[INFO] Reading training data')
wordlist = []
for news in training_data.data:
    wordlist.append(simple_preprocess(news))

print('[INFO] Creating Vocabulary')
model = Word2Vec(wordlist, size=10, window=4 ,workers=10)
print('[INFO] Training model')
model.train(wordlist, total_examples = len(wordlist), epochs=10)
word2vec_file = open('word2vec_model', 'wb')
pickle.dump(model, word2vec_file)
print('[INFO] Model Serialized and written to file')