import itertools
import time
import numpy as np
import pandas as pd
from skbio.diversity.alpha import shannon

import matplotlib.pyplot as plt
import seaborn as sns
import squarify

from kmer_approach import read_tcr_files

remote = True

if remote:
    path_to_tcr_dir = '/sample_sequences'
else:
    path_to_tcr_dir = '/home/ubuntu/Enno/gammaDelta/patient_data'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_data(sample, sample_class, v, col='read_count'):
    if v == 'all':
        data = sample[col]
        prefix = sample_class
    elif v == 'v1':
        data = sample[sample['v'] == 'TRDV1'][col]
        prefix = sample_class + '_TRDV1_'
    elif v == 'v2':
        data = sample[sample['v'] == 'TRDV2'][col]
        prefix = sample_class + '_TRDV2_'
    elif v == 'else':
        data = sample[(sample['v'] != 'TRDV1') & (sample['v'] != 'TRDV2')][col]
        prefix = sample_class + '_TRDVE_'
    else:
        raise ValueError('v parameter has to be \'all\', \'v1\', \'v2\', or \'else\'.')

    return data, prefix


def shannon_index(df, sample_class: str, v='all', kind='p'):
    result = []

    for ecrf in df.ecrf.unique():
        sample = df[df['ecrf'] == ecrf]
        data, prefix = get_data(sample, sample_class, v)

        result.append(shannon(data))

    return result


def simpson_index(df, sample_class: str, v='all', kind='p'):
    result = []

    for ecrf in df.ecrf.unique():
        sample = df[df['ecrf'] == ecrf]

        D = 0

        if kind == 'n':
            data, prefix = get_data(sample, sample_class, v)

            n = sum(sample.read_count)
            for n_i in data:
                numerator = n_i * (n_i - 1)
                denominator = n * (n - 1)

                D += numerator / denominator

        elif kind == 'p':
            data, prefix = get_data(sample, sample_class, v, col='freq')

            for p_i in data:
                D += p_i ** 2

            D = 1 - D

        else:
            raise ValueError('kind parameter has to be either \'n\' or \'p\'')

        result.append(D)

    return result


def morisita_index(df: pd.DataFrame, v: str):
    result = []

    if v == 'all':
        samples = [df[df['ecrf'] == ecrf] for ecrf in df.ecrf.unique()]
    elif v in ['v1', 'v2']:
        samples = [df[(df['ecrf'] == ecrf) & (df['v'] == 'TRDV' + v[-1])] for ecrf in df.ecrf.unique()]
    elif v == 'else':
        samples = [df[(df['ecrf'] == ecrf) & (df['v'] != 'TRDV1') & (df['v'] != 'TRDV2')] for ecrf in df.ecrf.unique()]
    else:
        raise ValueError('v parameter has to be \'all\', \'v1\', \'v2\', or \'else\'.')

    combinations = itertools.combinations(samples, 2)
    cs = itertools.combinations(samples, 2)
    N = len(list(cs))

    for ix, (x, y) in enumerate(combinations):  # x, y DataFrames

        if ix % len(samples) == 0: print(f'{(ix / N) * 100:.2f} %')
        X = x.read_count.to_list()
        X_seq = x.sequence.to_list()
        Y = y.read_count.to_list()
        Y_seq = y.sequence.to_list()

        if set(X_seq).intersection(set(Y_seq)) == {}:
            print('!')
            result.append(0)
            continue

        unique_species = np.unique(X_seq + Y_seq)
        numerator = 0
        de_numerator_x = 0
        de_numerator_y = 0

        for s_i in unique_species:
            x_i = x[x['sequence'] == s_i].read_count.to_list()[0] if s_i in x['sequence'].to_list() else 0
            y_i = y[y['sequence'] == s_i].read_count.to_list()[0] if s_i in y['sequence'].to_list() else 0

            numerator += x_i * y_i
            de_numerator_x += x_i ** 2
            de_numerator_y += y_i ** 2

        numerator *= 2
        denominator = (de_numerator_x / sum(X) ** 2 + de_numerator_y / sum(Y) ** 2) * sum(X) * sum(Y)

        C_h = numerator / denominator
        result.append(C_h)

    return result


def plot_tree_maps(df, sample_class: str, v: str = 'all', save=False):
    dir_tree_maps = '/home/enno/PycharmProjects/gamma_delta/plots/tree_maps/'

    for ecrf in df.ecrf.unique():
        sample = df[df['ecrf'] == ecrf]

        data, prefix = get_data(sample, sample_class, v)

        squarify.plot(data, alpha=0.8)
        name = f'{prefix}{ecrf}'
        plt.title(name)

        if save:
            plt.savefig(dir_tree_maps + f'{sample_class}/{v}/{name}.png')
        else:
            plt.show()

        plt.clf()


if __name__ == '__main__':
    t_0 = time.time()

    hd_df = read_tcr_files('HD', path_to_tcr_dir)
    bl_df = read_tcr_files('BL', path_to_tcr_dir)
    fu_df = read_tcr_files('FU', path_to_tcr_dir)

    print(f'\n{time.time() - t_0:.2f}s passed.')
