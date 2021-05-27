import time

import numpy as np
from Bio import SeqIO, pairwise2
from Bio.Align import substitution_matrices
from joblib import Parallel, delayed, dump
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap
import umap.plot
from networkit import *

def get_num_seq(f):
    num = len([1 for line in open(f) if line.startswith(">")])
    return num

"""
Available substitution matrices:

['BENNER22', 'BENNER6', 'BENNER74', 'BLOSUM45', 'BLOSUM50', 'BLOSUM62', 'BLOSUM80', 'BLOSUM90', 'DAYHOFF', 'FENG', 
 'GENETIC', 'GONNET1992', 'HOXD70', 'JOHNSON', 'JONES', 'LEVIN', 'MCLACHLAN', 'MDM78', 'NUC.4.4', 'PAM250', 'PAM30', 
 'PAM70', 'RAO', 'RISLER', 'SCHNEIDER', 'STR', 'TRANS']

"""

def parallel_fasta_to_distance_matrix(patient, substitution_matrix):
    """
    :param patient: FASTA-file
    :param substitution_matrix: String (e.g. "BLOSUM45", "GONNET1992")

    Calculates distance matrix from FASTA-file and saves as csv.
    """
    # TODO file = fr'/home/ubuntu/Enno/gammaDelta/patients/patient_{patient}.fasta'
    file = fr'/home/ubuntu/Enno/gammaDelta/patients/all_sequences.fasta'
    if patient == 0:
        file = fr'/home/ubuntu/Enno/gammaDelta/patients/all_sequences.fasta'

    n = get_num_seq(file)

    output_filename_memmap = r'/home/ubuntu/Enno/gammaDelta/joblib_memmap/output_memmap'
    output = np.memmap(output_filename_memmap, dtype=float, shape=(n, n), mode='w+')

    Parallel(n_jobs=28, verbose=50)(delayed(pairwise_score)(seqA, substitution_matrix, output, patient)
                                    for seqA in enumerate(SeqIO.parse(file, "fasta")))

    dump(output, fr'/home/ubuntu/Enno/gammaDelta/distance_matrices/ALL_SEQUENCES_{substitution_matrix}_DISTANCE_MATRIX')


def pairwise_score(seqa, substitution_matrix, output, patient):
    matrix = substitution_matrices.load(substitution_matrix)
    # TODO file = fr'/home/ubuntu/Enno/gammaDelta/patients/patient_{patient}.fasta'
    file = fr'/home/ubuntu/Enno/gammaDelta/patients/all_sequences.fasta'

    for seqb in enumerate(SeqIO.parse(file, "fasta")):
        res_ = pairwise2.align.globalds(seqa[1].seq, seqb[1].seq, matrix, -10, -0.5, score_only=True)
        output[seqa[0]][seqb[0]] = res_


def calculate_distance_matrices():
    """
    Computes distance matrices for a batch of patients
    and a batch of substitution matrices.
    """
    sm_batch = ["BLOSUM45", "BLOSUM80", "GONNET1992"]
    p_batch = range(1, 30)
    for sm in sm_batch:
        print("Current substitution matrix: ", sm)
        for p in p_batch:
            print("Working on patient ", str(p))
            parallel_fasta_to_distance_matrix(p, sm)


if __name__ == '__main__':

    for matrix in ['BLOSUM45', 'BLOSUM80', 'GONNET1992']:
        toc = time.time()
        parallel_fasta_to_distance_matrix_v2(0, matrix)
        tic = time.time()
        dump(f"{matrix} time elapsd: ", str(tic - toc))
