# ------------ Computes attention for each embedded entity e(i) ----------#

# Importing custom encoder for encoding sentences
import encoder
from encoder import encode_sentence
from GlobalHelpers import get_vector_based_on_word_index
from GlobalHelpers import get_absolute_path
from GlobalConfiguration import ENTITY_RETRIEVAL_CONFIG
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



def calculate_sum_of_exponents(entity_id_file, entity_vector_file, context_vector):
    sum_of_exponents = 0
    entity_num=1
    for vector in entity_vector_file:
        entity_vectors = vector.split("\t")
        entity_vectors.remove('\n')
        entity_vectors = [float(e) for e in entity_vectors]
        entity_vector, context_vector = pad_vectors(entity_vectors, context_vector)
        dot_prod = dot_product(entity_vector, context_vector)
        iteration_term = exponentiate(dot_prod)
        sum_of_exponents = sum_of_exponents + iteration_term
        entity_num = entity_num + 1
    #print('[INFO] SOE for '+str(entity_num))
    return sum_of_exponents

def calculate_attention(context_vector, entity_id_file, entity_vector_file, SUM_OF_EXPONENTS):
    entity_num = 1
    attention_headers = ['entityId', 'alpha_attention']
    csv_list = []
    file = open(ENTITY_RETRIEVAL_CONFIG().get_attention_file_path(), 'w')
    csv_list.append(attention_headers)
    for vector in entity_vector_file:
        entity_vector = vector.split('\t')
        entity_vector.remove('\n')
        entity_vector = [float(v) for v in entity_vector]
        entity_vector, context_vector = pad_vectors(entity_vector, context_vector)
        numerator = dot_product(entity_vector, context_vector)
        numerator = exponentiate(numerator)
        if SUM_OF_EXPONENTS != 0:
            alpha = numerator / SUM_OF_EXPONENTS
        else:
            alpha = 0
        csv_list.append([entity_num, alpha])
        entity_num = entity_num + 1
        #print('Attention Calculated for '+ str(entity_num))
    print('[INFO] Writing entity attentions into file')
    writer = csv.writer(file)
    writer.writerows(csv_list)


# ------------- Driver -------------#

def entity_attention_generator(context_vector):
    entity_id_file = open(get_absolute_path('/entity2id.txt'),'r')
    entity_vector_file = open(get_absolute_path('/entity2vec.bern'),'r')
    SUM_OF_EXPONENTS = calculate_sum_of_exponents(entity_id_file, entity_vector_file, context_vector )
    print('[INFO] Calculating attention')
    entity_id_file = open(get_absolute_path('/entity2id.txt'),'r')
    entity_vector_file = open(get_absolute_path('/entity2vec.bern'),'r')
    calculate_attention(context_vector, entity_id_file, entity_vector_file,SUM_OF_EXPONENTS)
    print('[INFO] Attention file generated for the specified context vector')


entity_attention_generator(encode_sentence('Samsung'))