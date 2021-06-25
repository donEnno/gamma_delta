import os
import numpy as np
from Bio.SeqIO.FastaIO import FastaIterator


def count_frequency_for_one_patient(patient_list: list, aa_sequences: list, actual, num: int):
    """
    :param patient_list: List of tuples (s, c), where s is #sequence and c is its assigned community.
    :param aa_sequences: List of all SeqIO objects.
    :param num: Total number of communities.
    :return: List of absolute frequencies for one patient.
    """
    frequency = np.zeros(num)

    for aa, (s, c) in zip(aa_sequences, patient_list):
        if frequency[c] == 0:
            frequency[c] = 1
        else:
            frequency[c] += 1
        actual[c].append(aa)

    return frequency, actual


def patient_get_num_sequences(patient_type: str):
    """
    :param patient_type: Either 'BL', 'FU', 'HD' or a combination of those, eg 'BLHD'
    :return: Number of sequences contained in patient's #num dataset.
    """
    num = []
    path = fr'/home/ubuntu/Enno/gammaDelta/sequence_data/{patient_type}_fasta/'

    num_patients = len(os.listdir(path))
    num_patients = range(1, num_patients+1)

    for patient_num in num_patients:
        file = fr'{path}{patient_type}_PATIENT_{patient_num}.fasta'
        num.append(len([1 for line in open(file) if line.startswith(">")]))

    return num


def fasta_to_seqio_list():
    """
    :return: List of SeqIO objects.
    """

    # TODO Make generic
    file = fr"/home/ubuntu/Enno/gammaDelta/sequence_data/BLHD_fasta/BLHD_ALL_SEQUENCES.fasta"

    aa_sequences = []
    with open(file) as handle:
        c = 0
        for record in FastaIterator(handle):
            aa_sequences.append(record)

    return aa_sequences


def get_frequencies(partition, patient_types, absolute_toggle=True):
    """
    Returns a np.array of shape (num_patients, num_communities) with community frequencies.
    :param partition: Result of NetworKit-based Louvain algorithm.
    :param patient_types: Either 'BL', 'FU', 'HD' or a combination of those, eg 'BLHD'
    :param absolute_toggle: Set False to return relative frequencies.

    :return Either absolute or relative cluster frequency and the clustered sequences.
    """

    absolute = []
    relative = []
    num_seq_per_patient = []

    list_of_sequences = fasta_to_seqio_list()

    for typ in patient_types:
        num_seq_per_patient.extend(patient_get_num_sequences(typ))

    total_num_seq = sum(num_seq_per_patient)
    num_partitions = len(np.unique(partition))

    # Initialize containers for SeqIO objects.
    actual = [[] for i in range(num_partitions)]

    # Sometimes partition is bugged and not every value is assigned. Therefore one needs to map them properly.
    if max(partition) != num_partitions:
        dic = dict(zip(np.unique(partition), range(num_partitions)))
        partition = [dic[i] for i in partition]

    # Create tuples (s, c) where s is the sequence' index and c its assigned cluster.
    partition = list(zip(range(total_num_seq), partition))

    upper = 0
    for i in num_seq_per_patient:
        lower = upper
        upper += i

        # Split partition and list_of_sequences in [lower:upper] where [l:u] is the range of sequences for one patient.
        temp_freq, actual = count_frequency_for_one_patient(partition[lower:upper], list_of_sequences[lower:upper], actual, num_partitions)
        temp_sum = sum(temp_freq)

        absolute.append(temp_freq)
        relative.append(temp_freq/temp_sum)

    if absolute_toggle:
        return np.array(absolute), actual

    return np.array(relative), actual

