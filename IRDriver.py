# A single driver script for information retrieval
import EntityRetrieval, EntityRetrievalDriver
import EntityAttentionGenerator, RelationAttentionGenerator
import RelationRetrieval, RelationRetrievalDriver
import encoder
import numpy as np
import GlobalHelpers
import mid2name
import pickle
from sklearn.datasets import fetch_20newsgroups

def extract_tripelt_mid_version(sentence, debug=False):
    encoded_sentence = encoder.encode_sentence(sentence)
    EntityAttentionGenerator.entity_attention_generator(encoded_sentence)
    RelationAttentionGenerator.relation_attention_generator(encoded_sentence)
    head_entity_vector = EntityRetrieval.entity_retrieve(open(GlobalHelpers.get_absolute_path('/entity2id.txt'),'r'), open(GlobalHelpers.get_absolute_path('/entity2vec.bern'),'r'))
    relation_vector = RelationRetrieval.relation_retrieve(open(GlobalHelpers.get_absolute_path('/relation2id.txt'),'r'), open(GlobalHelpers.get_absolute_path('/relation2vec.bern'),'r'))
    head_entity_mid = EntityRetrievalDriver.entity_retrieval_driver(head_entity_vector)
    relation = RelationRetrievalDriver.relation_retrieval_driver(relation_vector)
    head, relation = GlobalHelpers.pad_numpy_vectors(head_entity_mid, relation)
    tail_entity = np.add(head_entity_vector, relation_vector)
    tail_entity_mid = EntityRetrievalDriver.entity_retrieval_driver(tail_entity)
    print(head_entity_mid, relation, tail_entity_mid) 

def extract_triplet(sentence, debug=False):
    encoded_sentence = encoder.encode_sentence(sentence)
    EntityAttentionGenerator.entity_attention_generator(encoded_sentence)
    RelationAttentionGenerator.relation_attention_generator(encoded_sentence)
    head_entity_vector = EntityRetrieval.entity_retrieve(open(GlobalHelpers.get_absolute_path('/entity2id.txt'),'r'), open(GlobalHelpers.get_absolute_path('/entity2vec.bern'),'r'))
    relation_vector = RelationRetrieval.relation_retrieve(open(GlobalHelpers.get_absolute_path('/relation2id.txt'),'r'), open(GlobalHelpers.get_absolute_path('/relation2vec.bern'),'r'))
    head_entity_mid = EntityRetrievalDriver.entity_retrieval_driver(head_entity_vector)
    relation = RelationRetrievalDriver.relation_retrieval_driver(relation_vector)
    head, relation = GlobalHelpers.pad_numpy_vectors(head_entity_mid, relation)
    tail_entity = np.add(head_entity_vector, relation_vector)
    tail_entity_mid = EntityRetrievalDriver.entity_retrieval_driver(tail_entity)
    relevant_triple = (mid2name.mid2name(head_entity_mid), relation, mid2name.mid2name(tail_entity_mid))
    if debug:
        print(relevant_triple)
    return relevant_triple
  

def prepare_testing_data(limit=100,offset=0):
    test_data_with_triplets = []
    newsgroups_test = fetch_20newsgroups(subset='test')
    newsgroups_test_data =  newsgroups_test.data
    for i in range(offset, limit):
        print('####### Extracting for '+str(i)+" #########")
        print(newsgroups_test_data[i])
        current = newsgroups_test_data[i]
        kt = extract_triplet(newsgroups_test_data[i], debug=True)
        print(kt)
        for j in range(0, len(kt)):
            current =  current + kt[j]
        test_data_with_triplets.append(current)
    file = open('test_data_kt.pickle', 'wb')
    pickle._dump(test_data_with_triplets,file)
    print('[OK] Test data Prepared')

prepare_testing_data()