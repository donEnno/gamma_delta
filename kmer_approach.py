import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA
import seaborn as sns

# GLOBAL PATH VARS
path_to_tcr_dir = '/home/ubuntu/Enno/gammaDelta/patient_data'
path_to_dummy = 'dummy.fasta'


# METHODS FOR READING TCR DATA
def read_dummy_fasta(path):
    df = pd.DataFrame(columns=['ecrf', 'sequence', 'v', 'count', 'freq'])
    f = open(path, 'r')

    while True:
        header = f.readline().split('_')
        seq = f.readline().strip()

        if not seq:  # EOF
            break

        ecrf = header[0][1:]
        freq = float(header[2])
        count = int(header[3])
        v = header[4].strip()

        df.loc[len(df)] = [ecrf, seq, v, count, freq]

    return df


def read_tcr_files(patient_class, path, v=False):
    # Takes two minutes to read all files.
    df = pd.DataFrame(columns=['ecrf', 'sequence', 'v', 'read_count', 'freq'])

    path_to_data = path + f'/{patient_class}'
    files = os.listdir(path_to_data)

    files_to_exclude = ['FU_VDJTOOLS_.1003_FU_4-2-TCRD_S37_L001_R1.txt',
                        'FU_VDJTOOLS_.1004_FU_4-4-TCRD_S39_L001_R1.txt',
                        'BL_VDJTOOLS_.1003_BL_4-1-TCRD_S36_L001_R1.txt',
                        'BL_VDJTOOLS_.1004_BL_4-3-TCRD_S38_L001_R1.txt']

    for filename in files:

        if filename in files_to_exclude:
            continue

        ecrf = get_ecrf(filename)
        if v: print(ecrf)

        path_to_file = path_to_data + f'/{filename}'

        with open(path_to_file, 'r') as f:
            lines = f.readlines()

            for line in lines:
                line = line.split()

                if line[0] == '"cdr3aa"':
                    continue
                else:
                    sequence = line[0].replace('"', '')
                    v = line[1].replace('"', '')
                    count = int(line[2])
                    freq = float(line[3])

                    df.loc[len(df)] = [ecrf, sequence, v, count, freq]

    return df


def get_ecrf(filename: str):
    ecrf = ''

    if filename.startswith('BL') or filename.startswith('FU'):
        if 'Copy' in filename:
            ecrf = filename[19:23]
        else:
            ecrf = filename[13:17]

    elif filename.startswith('HD'):
        ecrf = filename.split('_')[-3]

    return ecrf


# METHODS FOR GENERATING KMERS
def split_n_way(a, n=3):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def get_kmers(df, size=5, use_count=True):
    list_of_kmer_counts = []

    for p in df.ecrf.unique():
        print('P', p)
        dic = {}
        sequences = df[df.ecrf == p].sequence
        read_counts = df[df.ecrf == p].read_count
        # print(len(sequences))
        # print(len(read_counts))

        for seq, read_c in zip(sequences, read_counts):

            len_s = list(range(len(seq) - size + 1))
            for ix in len_s:
                s, m, e = split_n_way(len_s)

                if ix in s:
                    suffix = '_s'
                elif ix in m:
                    suffix = '_m'
                elif ix in e:
                    suffix = '_e'
                else:
                    raise ValueError('A problem assigning the kmer position occurred.')

                k_mer = seq[ix:ix+size] + suffix

                if k_mer in dic:
                    if use_count:
                        dic[k_mer] += read_c
                    else:
                        dic[k_mer] += 1
                else:
                    if use_count:
                        dic[k_mer] = read_c
                    else:
                        dic[k_mer] = 1
        # print(len(dic))
        list_of_kmer_counts.append(dic)

        # MANUAL CURATION
        # print(sorted(dic.items(), key=lambda x: x[1], reverse=True))
        # names = list(dic.keys())
        # values = list(dic.values())
        # n_values = [v/len(values) for v in values]
        # all_n_values.extend(n_values)
        # print(max(n_values))

    return list_of_kmer_counts


def fill_dicts(list_of_dictionaries):
    """
    Takes two minutes for all generated kmers.

    :param list_of_dictionaries:
    :return: list of tuples that are equal in keys but differ in values
    """
    updated_kmer_counts = []

    for target in list_of_dictionaries:

        for query in list_of_dictionaries:

            if target == query:
                continue

            else:
                for kmer in query.keys():

                    if kmer in target.keys():
                        continue
                    else:
                        target[kmer] = 0

        target = sorted(target.items(), key=lambda x: x[0], reverse=True)
        updated_kmer_counts.append(target)

    return updated_kmer_counts


def build_p_k_mat(total_k_mer_counts):
    """
    :param total_k_mer_counts: All entries should have the same length.
    :return:
    """
    # TODO Restrain order in which the sub-samples are given to the method.
    pkm = np.zeros((len(filled_total_counts[0]), len(filled_total_counts)))
    for ix in range(pkm.shape[1]):
        pkm[:, ix] = list(zip(*filled_total_counts[ix]))[1]

    return pkm

def get_top_values(list_of_counts, d=10):
    out = []
    for count_list in list_of_counts:
        count_sum = sum([item[1] for item in count_list])
        count_list = [(kmer, count / count_sum) for kmer, count in count_list]
        n = len(count_list)
        first = int(n / d)
        out.append(count_list[:first])
    return out


if __name__ == '__main__':
    t_0 = time.time()

    hd_df = read_tcr_files('HD', path_to_tcr_dir)
    hd_counts = get_kmers(hd_df)

    bl_df = read_tcr_files('BL', path_to_tcr_dir)
    bl_counts = get_kmers(bl_df)

    fu_df = read_tcr_files('FU', path_to_tcr_dir)
    fu_counts = get_kmers(fu_df)

    h_sorted = [sorted(dic.items(), key=lambda x: x[1], reverse=True) for dic in hd_counts]
    b_sorted = [sorted(dic.items(), key=lambda x: x[1], reverse=True) for dic in bl_counts]
    f_sorted = [sorted(dic.items(), key=lambda x: x[1], reverse=True) for dic in fu_counts]

    top_h_sorted = get_top_values(h_sorted, 100)
    top_b_sorted = get_top_values(b_sorted, 100)
    top_f_sorted = get_top_values(f_sorted, 100)

    top_h_sorted_dicts = [dict(zip(list(zip(*h))[0], list(zip(*h))[1])) for h in top_h_sorted]
    top_b_sorted_dicts = [dict(zip(list(zip(*h))[0], list(zip(*h))[1])) for h in top_b_sorted]
    top_f_sorted_dicts = [dict(zip(list(zip(*h))[0], list(zip(*h))[1])) for h in top_f_sorted]

    total_counts = []
    total_counts.extend(top_h_sorted_dicts)
    total_counts.extend(top_b_sorted_dicts)
    total_counts.extend(top_f_sorted_dicts)

    filled_total_counts = fill_dicts(total_counts)

    PKM = build_p_k_mat(filled_total_counts)

    print(f'\n{time.time() - t_0:.2f}s passed.')
