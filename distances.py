# Default
import time
import joblib
import numpy as np
from joblib import Parallel, delayed, dump

# Biopython
import Bio.Seq
from Bio import SeqIO, pairwise2
from Bio.Align import substitution_matrices
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from Bio.Phylo.TreeConstruction import DistanceCalculator
from Bio import AlignIO
# from Bio.Align.Applications import ClustalOmegaCommandline


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                Available substitution matrices in Biopython                                         #
#                                                                                                                     #
#   'BENNER22', 'BENNER6', 'BENNER74', 'BLOSUM45', 'BLOSUM50', 'BLOSUM62', 'BLOSUM80', 'BLOSUM90', 'DAYHOFF', 'FENG'  #
#  'GENETIC', 'GONNET1992', 'HOXD70', 'JOHNSON', 'JONES', 'LEVIN', 'MCLACHLAN', 'MDM78', 'NUC.4.4', 'PAM250', 'PAM30' #
#                            'PAM70', 'RAO', 'RISLER', 'SCHNEIDER', 'STR', 'TRANS'                                    #
#                                                                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_num_seq(file):
    """
    :param file: Fasta-file
    :return: Number of sequences in a Fasta-file.
    """
    num = len([1 for line in open(file) if line.startswith(">")])
    return num


def parallel_fasta_to_distance_matrix(patient: str, substitution_matrix: str, go: int, ge: float):
    """
    Calculates distance matrix from FASTA-file in a parallel way and joblib.dumps it to a file.

    :param go: GapOpening penalty for pairwise alignment score. Must be a positive.
    :param ge: GapExtension penalty for pairwise alignment score. Must be positive.
    :param patient: Name of desired patient group: 'BLHD', 'BL', 'HD', 'FU'
    :param substitution_matrix: String (e.g. "BLOSUM45", "GONNET1992")
    """
    # TODO fr"/home/ubuntu/Enno/gammaDelta/sequence_data/HD_fasta/HD_PATIENT_3.fasta"
    file = fr"/home/ubuntu/Enno/gammaDelta/sequence_data/{patient}_fasta/{patient}_ALL_SEQUENCES.fasta"

    n = get_num_seq(file)

    output_filename_memmap = '/home/ubuntu/Enno/gammaDelta/joblib_memmap/output_memmap'
    output = np.memmap(output_filename_memmap, dtype=float, mode='w+', shape=(n, n))

    Parallel(n_jobs=-1, verbose=50)(delayed(pairwise_score)(patient, substitution_matrix, go, ge, seqA, output)
                                    for seqA in enumerate(SeqIO.parse(file, "fasta")))

    # TODO dump(output, fr'/home/ubuntu/Enno/mnt/volume/distance_matrices/{patient}_ALL_SEQUENCES_{substitution_matrix}_DISTANCE_MATRIX_{go}_{ge}')
    dump(output,
         fr'/home/ubuntu/Enno/mnt/volume/distance_matrices/BLFUHD_TEST')


def pairwise_score(patient: int, substitution_matrix: str, go: int, ge: float, seqa: Bio.Seq.Seq, output):
    """
    Calculates the pairwise score of seqa to every other sequence in the patient dataset.

    :param go: GapOpening penalty for pairwise alignment score. Must be a positive.
    :param ge: GapExtension penalty for pairwise alignment score. Must be positive.
    :param patient: Number of desired patient. 0 for all patients.
    :param substitution_matrix: Name of the substitution matrix to be used.
    :param seqa: Sequence to be scored against.
    :param output: np.memmap which translates to a np.ndarray
    """

    matrix = substitution_matrices.load(substitution_matrix)

    # TODO file = fr"/home/ubuntu/Enno/gammaDelta/sequence_data/HD_fasta/HD_PATIENT_3.fasta"
    file = fr"/home/ubuntu/Enno/gammaDelta/sequence_data/{patient}_fasta/{patient}_ALL_SEQUENCES.fasta"

    # TODO gloabds , matrix, -go, -ge, score_only=True
    for seqb in enumerate(SeqIO.parse(file, "fasta")):
        # for seqb in enumerate(SeqIO.parse(file, "fasta")):
        #     if seqa[0] % 1000 == 0:
        #         print(seqa[0])
        #     res_ = pairwise2.align.globalxx(seqa[1].seq, seqb[1].seq, score_only=True)
        #     output[seqa[0]][seqb[0]] = res_
        if seqa[0] > seqb[0]:
            res_ = pairwise2.align.globalxx(seqa[1].seq, seqb[1].seq, score_only=True)
            output[seqa[0]][seqb[0]] = res_
        else:
            continue


def calculate_distance_matrices():
    """
    Computes distance matrices for all patients
    and a batch of substitution matrices.
    """

    parallel_fasta_to_distance_matrix("BLHD", "BLOSUM45", 10, 0.5)


if __name__ == '__main__':
    parallel_fasta_to_distance_matrix("BLFUHD", "BLOSUM45", 10, 0.5)
