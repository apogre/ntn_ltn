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