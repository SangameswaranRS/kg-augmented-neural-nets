import gensim
import pickle
from gensim.models import Word2Vec

word = input('Enter word: ')
word2vec_file = open('word2vec_model','rb')
word2vec_model = pickle.load(word2vec_file)
try:
    print((word2vec_model.wv[word]))
    print(word2vec_model.wv.most_similar(positive=word, topn=6))
except Exception as E:
    print('Word not found in dictionary')