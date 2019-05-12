# Driver code for Entity Retrieval

# Distance metric used: Cosine similarity

from sklearn.metrics.pairwise import cosine_similarity
import GlobalHelpers
import numpy as np


# -------------- Helper methods --------------- #

def find_entity_mid(entity_vector_file, entity_id):
    # Return the corresponding entity mid
    current_eid = 1
    for line in entity_vector_file:
        if entity_id == current_eid:
            parsed_line = line.split('\t')
            return parsed_line[0]
        current_eid = current_eid + 1
    return None


def vector_with_max_cosine_sim(entity_vector_file, entity_vector, debug = True):
    max_similarity_extent = -1
    max_similar_eid = -1
    line_num = 1
    for line in entity_vector_file:
        vector = line.split("\t")
        vector.remove('\n')
        vector = [float(v) for v in vector]
        vector_array = np.array(vector)
        vector1, vector2 = GlobalHelpers.pad_numpy_vectors(vector_array, entity_vector)
        similarity_extent = cosine_similarity(vector1.reshape(1,-1), vector2.reshape(1,-1))
        if debug:
            print('[DEBUG] Similarity Extent for eid '+ str(line_num)+' is : '+ str(similarity_extent))
        if similarity_extent[0][0] > max_similarity_extent:
            max_similarity_extent = similarity_extent
            max_similar_eid = line_num
        line_num = line_num + 1
    return max_similar_eid


# --------------- Driver ------------ #

def entity_retrieval_driver(entity_vector):
    entity_id_file = open(GlobalHelpers.get_absolute_path('/entity2id.txt'),'r')
    entity_vector_file = open(GlobalHelpers.get_absolute_path('/entity2vec.bern'),'r')
    most_relevant_eid = vector_with_max_cosine_sim(entity_vector_file, entity_vector, debug=False)
    return find_entity_mid(entity_id_file, most_relevant_eid)