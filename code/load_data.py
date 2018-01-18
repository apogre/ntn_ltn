import numpy as np


def get_dictionary(file_name):
    """ Read and split data linewise """
    file_object = open(file_name, 'r')
    data = file_object.read().splitlines()
    """ Initialize dictionary to store the mapping """
    dictionary = {}
    index = 0
    for entity in data:
        """ Assign unique index to every entity """
        dictionary[entity] = index
        index += 1
    """ Number of entries in the data file """
    num_entries = index
    return dictionary, num_entries


def get_training_data(file_name, entity_dictionary, relation_dictionary):
    """ Read and split data linewise """
    file_object = open(file_name, 'r')
    data = file_object.read().splitlines()
    """ Initialize training data as an empty matrix """
    num_examples = len(data)
    training_data = np.empty((num_examples, 3), dtype=int)
    index = 0
    for line in data:
        """ Obtain relation example text by splitting line """
        entity1, relation, entity2 = line.split()

        """ Assign indices to the obtained entities and relation """
        training_data[index, 0] = entity_dictionary[entity1]
        training_data[index, 1] = relation_dictionary[relation]
        training_data[index, 2] = entity_dictionary[entity2]
        index += 1
    return training_data, num_examples


def get_feed(data_batch, num_relations):
    for i in range(num_relations):
        """ Make a list of examples for the 'i'th relation """
        rel_i_list = (data_batch['rel'] == i)
        num_rel_i = np.sum(rel_i_list)  # number of triples with relation i from training data
        """ Get entity lists for examples of 'i'th relation """
        e1 = data_batch['e1'][rel_i_list]
        e2 = data_batch['e2'][rel_i_list]
        e3 = data_batch['e3'][rel_i_list]
        """ Get entity vectors for examples of 'i'th relation """
        entity_vectors_e1 = entity_vectors[:, e1.tolist()]
        entity_vectors_e2 = entity_vectors[:, e2.tolist()]
        entity_vectors_e3 = entity_vectors[:, e3.tolist()]
        """ Choose entity vectors and lists based on 'flip' """

        if flip:
            entity_vectors_e1_neg = entity_vectors_e1
            entity_vectors_e2_neg = entity_vectors_e3
            e1_neg = e1
            e2_neg = e3
        else:
            entity_vectors_e1_neg = entity_vectors_e3
            entity_vectors_e2_neg = entity_vectors_e2
            e1_neg = e3
            e2_neg = e2
