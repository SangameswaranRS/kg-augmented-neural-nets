# Retreive Relation based on the attention values generated from RelationAttentionGenerator.py

import GlobalConfiguration, GlobalHelpers
import pandas as pd
import numpy as np


# --------------- Helpers ------------------#

def retreive_attention(relation_id, relation_attention_file_path):
    df = pd.read_csv(relation_attention_file_path)
    df = df['alpha_attention']
    return float(df[[relation_id]])

def relation_retrieve(relation_id_file, relation_vector_file):
    relation_attention_file_path = GlobalConfiguration.RELATION_RETRIEVAL_CONFIG().get_attention_file_path()
    relation_id = 1
    sum_vector = np.array([])
    for vector in relation_vector_file:
        relation_vector = vector.split('\t')
        relation_vector.remove('\n')
        relation_vector = [float(v) for v in relation_vector]
        e_i = relation_vector
        alpha_i = retreive_attention(relation_id-1, relation_attention_file_path)
        scaled_vector = GlobalHelpers.scalar_prod_vector(e_i, alpha_i)
        sum_vector, scaled_vector = GlobalHelpers.pad_numpy_vectors(sum_vector,scaled_vector)
        sum_vector = np.add(sum_vector, scaled_vector)
        #print('alpha_i*r_i Computed for :'+str(relation_id))
        relation_id = relation_id + 1
    return sum_vector

# ----------- Driver test-----------#
print(relation_retrieve(open(GlobalHelpers.get_absolute_path('/relation2id.txt'),'r'), open(GlobalHelpers.get_absolute_path('/relation2vec.bern'),'r')))
