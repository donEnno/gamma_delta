import os

import numpy as np
from projection import calculate_partitions

# # # # # # # # # # # # # # # # RESULT OF NX BASED METHOD # # # # # # # # # # # # # # # # # # #
# parti_on = load('/home/ubuntu/Enno/gammaDelta/partition/all_patients_BLOSUM45_communities')
# parti_on = load("/home/ubuntu/Enno/gammaDelta/partition/nx/patient_10_BLOSUM45_resolution=0.875_communities")
# parti_on = plot_louvain(0, 'BLOSUM45', netx=False, gamma=1.045)
# print(parti_on)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def count_frequency(patient_list: list, num: int):
    """
    :param patient_list: List of tuples (s, c), where s is #sequence and c is its assigned community.
    :param num: Total number of communities.
    :return: List of absolute frequencies for one patient.
    """
    frequency = np.zeros(num)

    for s, c in patient_list:
        if frequency[c] == 0:
            frequency[c] = 1
        else:
            frequency[c] += 1

    return frequency


def patient_get_num_sequences(patient_type: str):
    """
    :param patient_type:
    :return: Number of sequences contained in patient's #num dataset.
    """
    num = []
    path = fr'/home/ubuntu/Enno/gammaDelta/sequence_data/{patient_type}_fasta/'
    num_patients = len(os.listdir(path))

    print('From patient_get_num_sequences: \n'
          f'Number of patients for {patient_type} is ', num_patients)

    for i in range(1, num_patients+1):
        file = fr'{path}{patient_type}_PATIENT_{i}.fasta'
        num.append(len([1 for line in open(file) if line.startswith(">")]))

    print('From patient_get_num_sequences: \n'
          f'List of number of seq per patients for {patient_type} is ', num)

    return num


def get_frequencies(partition, patient_types, absolute_toggle=False):
    """
    Returns a np.array of shape (num_patients, num_communities) with community frequencies.
    :param partition: Result of NetworKit-based Louvain algorithm.
    :param patient_types:
    :param absolute_toggle: Set True to return absolute frequencies.
    """
    absolute = []
    relative = []
    num_seq_per_patient = []

    # num_seq_per_patient.extend(patient_get_num_sequences(typ) for typ in patient_types)
    for typ in patient_types:
        num_seq_per_patient.extend(patient_get_num_sequences(typ))

    print('From get_frequencies: \n'
          'Total num_seq_per_patient: ', num_seq_per_patient)

    total_num_seq = sum(num_seq_per_patient)
    print('From get_frequencies: \n'
          'total_num_seq: ', total_num_seq)

    partition = partition.getVector()
    num_partitions = len(np.unique(partition))
    partition = list(zip(range(total_num_seq), partition))

    upper = 0
    for i in num_seq_per_patient:
        lower = upper
        upper += i

        temp = count_frequency(partition[lower:upper], num_partitions)
        temp_sum = sum(temp)

        absolute.append(temp)
        relative.append(temp/temp_sum)

    if absolute_toggle:
        return np.array(absolute)

    return np.array(relative)


def get_cluster_membership(patient_list: list, num: int):
    """
    :param patient_list: List of tuples (s, c), where s is #sequence and c is its assigned community.
    :param num: Total number of communities.
    :return: List of absolute frequencies for one patient.
    """
    frequency = [[] for i in range(num)]

    for s, c in patient_list:
        frequency[c].append(s)

    return np.array(frequency)


def compute_overlap(partition_1):
    """
    Returns a np.array of shape (num_patients, num_communities) with community frequencies.
    :param partition_1: Result of nk-based Louvain algorithm.
    :param absolute_toggle: Set True to return absolute frequencies.
    """
    absolute = []
    relative = []

    num_patients = 29
    total_num_seq = 10711

    partition_1 = partition_1.getVector()
    num_partitions = len(np.unique(partition_1))
    partition_1 = list(zip(range(total_num_seq), partition_1))

    # partition_2 = partition_2.getVector()
    # num_partitions = len(np.unique(partition_2))
    # partition_2 = list(zip(range(total_num_seq), partition_2))

    temp = get_cluster_membership(partition_1, num_partitions)

    return np.array(temp)

