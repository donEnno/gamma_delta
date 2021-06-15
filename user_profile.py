from Bio.Align import substitution_matrices
from joblib import Parallel, delayed
from distances import parallel_fasta_to_distance_matrix, calculate_distance_matrices
import pstats
from pstats import SortKey
import line_profiler
from Bio import SeqIO, pairwise2
from distances import get_num_seq
import numpy as np


# @profile
def profile_fasta_to_distance_matrix(substitution_matrix: str, go: int, ge: float):
    """
    Calculates distance matrix from FASTA-file in a parallel way and joblib.dumps it to a file.

    :param go: GapOpening penalty for pairwise alignment score. Must be a positive.
    :param ge: GapExtension penalty for pairwise alignment score. Must be positive.
    :param patient: Name of desired patient group: 'BLHD', 'BL', 'HD', 'FU'
    :param substitution_matrix: String (e.g. "BLOSUM45", "GONNET1992")
    """
    # TODO fr"/home/ubuntu/Enno/gammaDelta/sequence_data/{patient}_fasta/{patient}_ALL_SEQUENCES.fasta"
    file = fr"/home/ubuntu/Enno/gammaDelta/sequence_data/HD_fasta/HD_PATIENT_1.fasta"
    matrix = substitution_matrices.load(substitution_matrix)
    n = get_num_seq(file)

    output_filename_memmap = '/home/ubuntu/Enno/gammaDelta/joblib_memmap/output_memmap'
    output = np.memmap(output_filename_memmap, dtype=float, mode='w+', shape=(n, n))
    i = 0
    for seqa in enumerate(SeqIO.parse(file, "fasta")):
        if i % 10 == 0:
            print(i)
        i += 1
        for seqb in enumerate(SeqIO.parse(file, "fasta")):
            res_ = pairwise2.align.globalds(seqa[1].seq, seqb[1].seq, matrix, -go, -ge, score_only=True)
            output[seqa[0]][seqb[0]] = res_

    # TODO dump(output, fr'/home/ubuntu/Enno/mnt/volume/distance_matrices/{patient}_ALL_SEQUENCES_{substitution_matrix}_DISTANCE_MATRIX_{go}_{ge}')
    # dump(output,
    #      fr'/home/ubuntu/Enno/mnt/volume/distance_matrices/TEST')


if __name__ == '__main__':
    profile_fasta_to_distance_matrix('BLOSUM45', 10, 0.5)

"""
filename = '/home/ubuntu/Enno/gammaDelta/stats/profile_stats.stats'
profile.run("pairing()", filename)
stats = pstats.Stats('/home/ubuntu/Enno/gammaDelta/stats/profile_stats.stats')

# Clean up filenames for the report
# stats.strip_dirs()

# Sort the statistics by the cumulative time spent in the function
print(20*'= ', 'SortKey.CUMULATIVE', 20*' =')
stats.sort_stats(SortKey.CUMULATIVE).print_stats(50)

print(20*'= ', 'SortKey.TIME', 20*' =')
stats.sort_stats(SortKey.TIME).print_stats(50)

print(20*'= ', 'SortKey.PCALLS', 20*' =')
stats.sort_stats(SortKey.PCALLS).print_stats(50)"""



