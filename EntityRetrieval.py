# Retreive entity based on the attention values calculated from EntityAttentionGenerator.py

import GlobalConfiguration, GlobalHelpers
import pandas as pd
import numpy as np


# --------------- Helpers ------------------#

def retreive_attention(entity_id, entity_attention_file_path):
    df = pd.read_csv(entity_attention_file_path)
    df = df['alpha_attention']
    return float(df[[entity_id]])

def entity_retrieve(entity_id_file, entity_vector_file):
    entity_attention_file_path = GlobalConfiguration.ENTITY_RETRIEVAL_CONFIG().get_attention_file_path()
    entity_id = 1
    sum_vector = np.array([])
    for vector in entity_vector_file:
        entity_vector = vector.split('\t')
        entity_vector.remove('\n')
        entity_vector = [float(v) for v in entity_vector]
        e_i = entity_vector
        alpha_i = retreive_attention(entity_id-1, entity_attention_file_path)
        scaled_vector = GlobalHelpers.scalar_prod_vector(e_i, alpha_i)
        sum_vector, scaled_vector = GlobalHelpers.pad_numpy_vectors(sum_vector,scaled_vector)
        sum_vector = np.add(sum_vector, scaled_vector)
        # print('alpha_i*e_i Computed for :'+str(entity_id))
        entity_id = entity_id + 1
    return sum_vector

# ----------- Driver test-----------#
print(entity_retrieve(open(GlobalHelpers.get_absolute_path('/entity2id.txt'),'r'), open(GlobalHelpers.get_absolute_path('/entity2vec.bern'),'r')))


