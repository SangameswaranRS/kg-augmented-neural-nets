# Encode the sentences to vectors based on random word embeddings assigned by DKRL


from GlobalConfiguration import DKRL_CONFIG
from GlobalHelpers import get_absolute_path
from nltk.tokenize import word_tokenize



# ------------------ Helper Methods -------------------#

def get_vector_based_on_word_index(file, line_offset=0):
    line_count = 1
    vector_line = ''
    for line in file:
        if line_count == line_offset:
            vector_line = line
            break
        line_count = line_count + 1
    word_vectors = vector_line.split('\t')
    if '\n' in word_vectors:
        word_vectors.remove('\n')
    #print(word_vectors)
    if vector_line == '':
        return []
    return [float(w) for w in word_vectors]


# ----------------Encoder Method -----------------#

def encode_sentence(sentence, word_file_path=get_absolute_path('/word2id.txt'), vector_file_path=get_absolute_path('/word2vec.bern')):
    # Returns a vector based on words in a sentence - Word Embeddings at  word2vec -DKRL
    word_file = open(word_file_path,'r')
    vector_file = open(vector_file_path, 'r')
    words_in_sentence = word_tokenize(sentence)
    words_in_file = []
    for line in  word_file:
        tokenized_line = line.split("\t")
        words_in_file.append(tokenized_line[0].lower())
    sentence_vector = []
    for word in words_in_sentence:
        if word.lower() in words_in_file:
            word_index = words_in_file.index(word.lower())
            word_vectors = get_vector_based_on_word_index(vector_file,line_offset=word_index)
            for num in word_vectors:
                sentence_vector.append(num)
    return sentence_vector
