# ------------- Computes attention for each embedded relation r(i) ------------- #

# Importing custom encoder for encoding sentences
import encoder
from encoder import encode_sentence
from GlobalHelpers import get_vector_based_on_word_index
from GlobalHelpers import get_absolute_path
from GlobalConfiguration import RELATION_RETRIEVAL_CONFIG
import numpy as np
import csv
import pandas as pd

# -------- Numpy Helper Utils --------# 

def exponentiate(vector):
    # computes e^x for each x in vector
    return np.exp(vector)

def dot_product(vector1, vector2):
    # Returns the scalar product of vector 1 and 2
    return np.dot(vector1, vector2,out=None)

def pad_vectors(vector1, vector2):
    # Utility Function to pad vectors to equal Length
    if len(vector1) > len(vector2):
        # pad Vector 2
        vector2 += [0]*(len(vector1)-len(vector2))
        return (vector1, vector2)
    else:
        # pad Vector 1
        vector1 +=[0]*(len(vector2)-len(vector1))
        return (vector1, vector2)


def calculate_sum_of_exponents(relation_id_file, relation_vector_file, context_vector):
    sum_of_exponents = 0
    relation_num=1
    for vector in relation_vector_file:
        relation_vectors = vector.split("\t")
        relation_vectors.remove('\n')
        relation_vectors = [float(e) for e in relation_vectors]
        relation_vector, context_vector = pad_vectors(relation_vectors, context_vector)
        dot_prod = dot_product(relation_vector, context_vector)
        iteration_term = exponentiate(dot_prod)
        sum_of_exponents = sum_of_exponents + iteration_term
        relation_num = relation_num + 1
    #print('[INFO] SOE for '+str(relation_num))
    return sum_of_exponents

def calculate_attention(context_vector, relation_id_file, relation_vector_file, SUM_OF_EXPONENTS):
    relation_num = 1
    attention_headers = ['entityId', 'alpha_attention']
    csv_list = []
    file = open(RELATION_RETRIEVAL_CONFIG().get_attention_file_path(), 'w')
    csv_list.append(attention_headers)
    for vector in relation_vector_file:
        relation_vector = vector.split('\t')
        relation_vector.remove('\n')
        relation_vector = [float(v) for v in relation_vector]
        relation_vector, context_vector = pad_vectors(relation_vector, context_vector)
        numerator = dot_product(relation_vector, context_vector)
        numerator = exponentiate(numerator)
        if SUM_OF_EXPONENTS != 0:
            alpha = numerator / SUM_OF_EXPONENTS
        else:
            alpha = 0
        csv_list.append([relation_num, alpha])
        relation_num = relation_num + 1
        #print('Attention Calculated for '+ str(relation_num))
    print('[INFO] Writing relation attention into file')
    writer = csv.writer(file)
    writer.writerows(csv_list)


# ------------- Driver -------------#

def relation_attention_generator(context_vector):
    relation_id_file = open(get_absolute_path('/relation2id.txt'),'r')
    relation_vector_file = open(get_absolute_path('/relation2vec.bern'),'r')
    SUM_OF_EXPONENTS = calculate_sum_of_exponents(relation_id_file, relation_vector_file, context_vector )
    print('[INFO] Calculating attention')
    relation_id_file = open(get_absolute_path('/relation2id.txt'),'r')
    relation_vector_file = open(get_absolute_path('/relation2vec.bern'),'r')
    calculate_attention(context_vector, relation_id_file, relation_vector_file,SUM_OF_EXPONENTS)
    print('[INFO] Attention file generated for the specified context vector')

relation_attention_generator(encode_sentence('Samsung'))


