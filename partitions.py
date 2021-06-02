import numpy as np
from joblib import load


# # # # # # # # # # # # # # # # RESULT OF NX BASED METHOD # # # # # # # # # # # # # # # # # # #
parti_on = load('/home/ubuntu/Enno/gammaDelta/partition/all_patients_BLOSUM45_communities')   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def patient_get_num_sequences(num: int):
    """
    :param num: Number of desired patient.
    :return: Number of sequences cointained in patient's #num dataset.
    """
    file = fr'/home/ubuntu/Enno/gammaDelta/patients/patient_{num}.fasta'
    num = len([1 for line in open(file) if line.startswith(">")])
    return num


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


def get_frequencies(partition: dict, absolute_toggle=False):
    """
    Returns a np.array of shape (num_patients, num_communities) with community frequencies.
    :param partition: Result of nx-based Louvain algorithm.
    :param absolute_toggle: Set True to return absolute frequencies.
    """
    absolute = []
    relative = []

    num_partitions = max(partition.values())+1
    partition = list(partition.items())

    num_patients = 29
    num_seq_per_patient = [patient_get_num_sequences(i) for i in range(1, num_patients+1)]

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
