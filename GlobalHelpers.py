# ----------- Global Helper methods for the project -------#
from GlobalConfiguration import DKRL_CONFIG
import numpy as np


def get_absolute_path(relative_path):
    return DKRL_CONFIG.DKRL_HOME  + str(relative_path)

def get_vector_based_on_word_index(file, line_offset=0):
    line_count = 1
    vector_line = ''
    for line in file:
        if line_count == line_offset:
            vector_line = line
            break
        line_count = line_count + 1
    word_vectors = vector_line.split('\t')
    word_vectors.remove('\n')
    return [float(w) for w in word_vectors]


def scalar_prod_vector(vector, scalar):
    return np.multiply(vector, scalar)

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

def pad_numpy_vectors(vector1, vector2):
    if len(vector1) > len(vector2):
        vector2 = np.pad(vector2, (0, len(vector1) - len(vector2)),mode='constant')
        return vector1, vector2
    else:
        vector1 = np.pad(vector1, (0, len(vector2)-len(vector1)),mode='constant')
        return vector1, vector2
