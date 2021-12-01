import math
import numpy as np
from Bio import SeqIO, pairwise2
from Bio.Align import substitution_matrices
from joblib import Parallel, delayed, dump

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                Available substitution matrices in Biopython                                         #
#                                                                                                                     #
#   'BENNER22', 'BENNER6', 'BENNER74', 'BLOSUM45', 'BLOSUM50', 'BLOSUM62', 'BLOSUM80', 'BLOSUM90', 'DAYHOFF', 'FENG'  #
#  'GENETIC', 'GONNET1992', 'HOXD70', 'JOHNSON', 'JONES', 'LEVIN', 'MCLACHLAN', 'MDM78', 'NUC.4.4', 'PAM250', 'PAM30' #
#                            'PAM70', 'RAO', 'RISLER', 'SCHNEIDER', 'STR', 'TRANS'                                    #
#                                                                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def parallel_fasta_to_distance_matrix(file, patient, substitution_matrix, gap_open, gap_extend):

    sequences = list(enumerate(SeqIO.parse(file, 'fasta')))
    number_of_sequences = len(sequences)

    memmap_directory = '/home/ubuntu/Enno/gammaDelta/joblib_memmap/hpc1_output_memmap'
    output = np.memmap(memmap_directory, dtype=float, mode='w+', shape=(number_of_sequences, number_of_sequences))

    batches = generate_batches_from_file(file)

    total_pairs = sum([s[0] for b in batches for s in b])
    print('Number of total pairs:', total_pairs)

    Parallel(n_jobs=-1, verbose=50)(delayed(compute_pairwise_scores)(file=file,
                                                                     substitution_matrix=substitution_matrix,
                                                                     gap_open=gap_open, gap_extend=gap_extend,
                                                                     batch=batch,
                                                                     output=output,
                                                                     total_pairs=total_pairs)
                                    for batch in batches)

    output_file = fr'/home/ubuntu/Enno/mnt/volume/distance_matrices/{patient}_{substitution_matrix}_{gap_open}_{gap_extend}_DM'
    dump(output, output_file)


def compute_pairwise_scores(file, substitution_matrix, gap_open, gap_extend, batch, output, total_pairs):

    matrix = substitution_matrices.load(substitution_matrix)
    sequences = list(enumerate(SeqIO.parse(file, 'fasta')))

    batch_pair_count = 0
    for i, s in batch:
        batch_pair_count += i

    print('Working on ', (batch_pair_count / total_pairs) * 100, '% of pairs')

    for seq_a in batch:
        for seq_b in sequences:
            if seq_a[0] > seq_b[0]:
                res_ = pairwise2.align.globalds(seq_a[1].seq, seq_b[1].seq, matrix, -gap_open, -gap_extend, score_only=True)
                output[seq_a[0]][seq_b[0]] = res_

            else:
                continue


def generate_batches_from_file(file):

    sequences_with_index = list(enumerate(SeqIO.parse(file, "fasta")))

    n_of_pairs = 0
    for i in sequences_with_index:
        n_of_pairs += i[0]

    # TODO Magic number, num of cores minus 1
    n_jobs = 51
    sequences_per_job = math.ceil(n_of_pairs/n_jobs)

    job_batches = []
    temp_batch = []
    temp_sum = 0

    for ix in sequences_with_index:
        if not temp_batch:
            temp_sum += ix[0]
            temp_batch.append(ix)
        elif temp_sum + ix[0] > sequences_per_job:
            temp_sum = ix[0]
            temp_batch.append(ix)
            job_batches.append(temp_batch)
            temp_batch = []
        else:
            temp_sum += ix[0]
            temp_batch.append(ix)

    job_batches.append(temp_batch)

    return job_batches

    # names = ['BENNER6', 'BENNER22', 'BENNER74', 'BLOSUM45', 'BLOSUM50', 'BLOSUM62', 'BLOSUM80', 'BLOSUM90', 'DAYHOFF',
    #         'FENG', 'GENETIC', 'GONNET1992', 'JOHNSON', 'JONES', 'MCLACHLAN', 'MDM78', 'PAM30', 'PAM70', 'PAM250', 'RAO',
    #         'RISLER', 'STR']

def main():

    path_to_blfuhd = '/home/ubuntu/Enno/gammaDelta/sequence_data/BLFUHD_fasta/BLFUHD_ALL_SEQUENCES.fasta'

    jobs_done = 0

    # TODO
    names = ['BLOSUM62']
    penalties = [(1, 0.1), (10, 0.5), (25, 1.0)]

    for go, ge in penalties:
        print('GO/GE:', go, ge)
        for sm in names:
            print('Substitution Matrix', sm)

            parallel_fasta_to_distance_matrix(file=path_to_blfuhd, patient='BLFUHD', substitution_matrix=sm, gap_open=go, gap_extend=ge)
            jobs_done += 1
            print(jobs_done, 'jobs done!')

if __name__ == '__main__':

    main()