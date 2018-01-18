import logictensornetworks as ltn
import tensorflow as tf
from load_data import get_dictionary, get_training_data
import numpy as np
import sys

data_set = "../data/"
num_iterations = 100
number_of_features = 100
batch_size = 200
corrupt_size = 10
config = tf.ConfigProto(device_count={'GPU': 1}, log_device_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))

print(""" Get entity and relation data dictionaries """)
entity_dictionary, num_entities = get_dictionary(data_set + 'entities.txt')
relation_dictionary, num_relations = get_dictionary(data_set + 'relations.txt')

print(""" Get training data using entity and relation dictionaries """)
training_data, num_training_examples = get_training_data(data_set + 'train.txt', entity_dictionary, relation_dictionary)

entity_pair = ltn.Domain(2*(number_of_features), label="entity_pair")
entity_pair_of_r = ltn.Domain(2*(number_of_features), label="entity_pair_in_r")
not_entity_pair_of_r = ltn.Domain(2*(number_of_features), label="entity_pair_not_in_r")


is_of_rel = {}
for r in relation_dictionary.keys():
    is_of_rel[r] = ltn.Predicate("is_of_rel_"+r, entity_pair, layers=5)


for i in range(num_iterations):
    # print(""" Create a training batch by picking up random samples from training data """)
    batch_indices = np.random.randint(num_training_examples, size=batch_size) #Randomly sample training batch

    data = dict()
    data['rel'] = np.tile(training_data[batch_indices, 1], (1, corrupt_size)).T
    existing_relations = []
    data['e1'] = np.tile(training_data[batch_indices, 0], (1, corrupt_size)).T

    data['e2'] = np.tile(training_data[batch_indices, 2], (1, corrupt_size)).T
    data['e3'] = np.random.randint(num_entities, size=(batch_size * corrupt_size, 1))

    print data
    init = tf.initialize_all_variables()
    sess = tf.Session(config=config)

    clauses_for_positive_examples = [ltn.Clause([ltn.Literal(True, is_of_rel[r], entity_pair_of_r)]) for r in existing_relations]
    clauses_for_negative_examples = [ltn.Clause([ltn.Literal(True, is_of_rel[r], not_entity_pair_of_r)]) for r in existing_relations]

    clauses = clauses_for_negative_examples + clauses_for_positive_examples

    KB = ltn.KnowledgeBase("kb_label", clauses, "models/")
    if i == 1:
        sess.run(init)
    if i > 1:
        KB.restore(sess)
