# Driver code for Relation Retrieval

# Distance metric used: Cosine similarity

from sklearn.metrics.pairwise import cosine_similarity
import GlobalHelpers
import numpy as np


# -------------- Helper methods --------------- #

def find_relation(relation_id_file, relation_id):
    # Return the corresponding relation
    current_rid = 1
    for line in relation_id_file:
        if relation_id == current_rid:
            parsed_line = line.split('\t')
            return parsed_line[0]
        current_rid = current_rid + 1
    return None


def vector_with_max_cosine_sim(relation_vector_file, relation_vector, debug = True):
    max_similarity_extent = -1
    max_similar_rid = -1
    line_num = 1
    for line in relation_vector_file:
        vector = line.split("\t")
        vector.remove('\n')
        vector = [float(v) for v in vector]
        vector_array = np.array(vector)
        vector1, vector2 = GlobalHelpers.pad_numpy_vectors(vector_array, relation_vector)
        similarity_extent = cosine_similarity(vector1.reshape(1,-1), vector2.reshape(1,-1))
        if debug:
            print('[DEBUG] Similarity Extent for eid '+ str(line_num)+' is : '+ str(similarity_extent))
        if similarity_extent[0][0] > max_similarity_extent:
            max_similarity_extent = similarity_extent
            max_similar_rid = line_num
        line_num = line_num + 1
    return max_similar_rid


# --------------- Driver ------------ #

def relation_retrieval_driver(relation_vector):
    relation_id_file = open(GlobalHelpers.get_absolute_path('/relation2id.txt'),'r')
    relation_vector_file = open(GlobalHelpers.get_absolute_path('/relation2vec.bern'),'r')
    most_relevant_rid = vector_with_max_cosine_sim(relation_vector_file, relation_vector, debug=False)
    return find_relation(relation_id_file, most_relevant_rid)
