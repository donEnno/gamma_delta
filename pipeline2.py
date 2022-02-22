import copy
import time
import os
import joblib
import numexpr
import numpy as np
import pandas as pd
from statistics import mode

from Bio import SeqIO

import networkit

import umap
import umap.plot

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import classification_report

from scipy.spatial import KDTree

os.environ['NUMEXPR_MAX_THREADS'] = '52'

path_to_dm = 'D:/enno/2022/hiwi/data/dm/pam/BLFUHD_PAM70_10_0.5_DM'
path_to_data = '/home/ubuntu/Enno/gammaDelta/patient_data/all/'


dm_root = '/home/ubuntu/Enno/mnt/volume/dm_in_use/'
b45 = dm_root + 'BLFUHD_BLOSUM45_1_0.1_DM'
b62 = dm_root + 'BLFUHD_BLOSUM62_10_0.5_DM'
pam70 = "/home/ubuntu/Enno/mnt/volume/dm_in_use/BLFUHD_PAM70_1_0.1_DM"


def write_fasta():
    with open(path_to_fasta, 'w') as dummy_file:
        for raw_file, prefix, pat_no in zip(os.listdir(path_to_data),
                                            66 * ['BL'] + 55 * ['FU'] + 29 * ['HD'],
                                            list(range(1, 67)) + list(range(1, 56)) + list(range(1, 30))):
            path_to_file = path_to_data + '/' + raw_file
            df = pd.read_csv(path_to_file, delimiter='\t')

            for ix, (seq, f, c, v) in enumerate(zip(df['cdr3aa'], df['freq'], df['count'], df['v'])):
                header = '>{}-{}-_sequence-{}_{}_{}_{}'.format(prefix, pat_no, ix + 1, f, c, v)
                dummy_file.write(header + '\n')
                dummy_file.write(seq + '\n')


def get_dm(path='dummy_dm', full=False):
    dm = joblib.load(path)
    if full:
        dm = (dm + dm.T)

    return dm


def shift_similarities_to_zero(similarity_matrix):
    minimum = similarity_matrix.min()
    similarity_matrix = similarity_matrix - minimum

    return similarity_matrix


def f(x):
    return 1/(x+1)


def similarities_to_distances(similarity_matrix):
    similarity_matrix = shift_similarities_to_zero(similarity_matrix)
    f_vec = np.vectorize(f)
    distance_matrix = f_vec(similarity_matrix)
    np.fill_diagonal(distance_matrix, 0)

    return distance_matrix


def get_dm_train_test(df, dm):
    patient_ids = np.unique([x[0] for x in df.index])
    response = np.array([1 if 'BL' in patient or 'FU' in patient else 0 for patient in patient_ids])

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

    for train_index, test_index in sss.split(patient_ids, response):
        train_index.sort()
        test_index.sort()

        X_train, X_test = patient_ids[train_index], patient_ids[test_index]
        y_train, y_test = response[train_index], response[test_index]

    train_ixs = [ix for patient_ixs in [get_patient_indices(tag, df) for tag in X_train] for ix in patient_ixs]
    test_ixs = [ix for patient_ixs in [get_patient_indices(tag, df) for tag in X_test] for ix in patient_ixs]

    train_dm = np.take(np.take(dm, train_ixs, axis=0), train_ixs, axis=1)
    test_dm = np.take(np.take(dm, test_ixs, axis=0), train_ixs, axis=1)

    train_df = df.iloc[train_ixs]
    test_df = df.iloc[test_ixs]

    return train_df, train_dm, y_train, test_df, test_dm, y_test


path_to_fasta = '/home/ubuntu/Enno/gammaDelta/patient_data/blfuhd.fasta'


def get_fasta_info(file=path_to_fasta):
    fasta = list(SeqIO.parse(file, 'fasta'))
    index = [(p, seq_no) for [p, seq_no, f, c, v], seq in [(record.id.split('_'), str(record.seq)) for record in fasta]]
    data = [(seq, f, c, v) for [p, n_seq, f, c, v], seq in [(record.id.split('_'), str(record.seq)) for record in fasta]]
    multi_index = pd.MultiIndex.from_tuples(index, names=['patient', 'seq_no'])

    df = pd.DataFrame(data, index=multi_index, columns=['sequence', 'freq', 'count', 'v'])

    return df


def get_graph(dm):
    t0 = time.time()
    m, _ = dm.shape
    g = networkit.Graph(m, weighted=True)
    mask_x, mask_y = np.mask_indices(m, np.tril, -1)
    masking_zip = zip(mask_x, mask_y, dm[mask_x, mask_y])

    for nodeA, nodeB, weight in masking_zip:
        if weight == 0:
            continue
        g.addEdge(nodeA, nodeB, weight)

    print('The graph construction took %.3f s' % (time.time()-t0))

    return g


def get_embedding(dm):
    reducer = umap.UMAP().fit_transform(dm)

    return reducer


def get_cluster(graph, gamma=1, n_cluster=4, affinity_mat=[], kind='louvain'):

    cluster_vector = []
    cluster_ids = []

    if kind not in ['louvain', 'leiden', 'spectral']:
        raise ValueError('\'kind\' has to be either \'louvain\', \'leiden\' or \'spectral\'')

    if kind == 'louvain':
        cluster = networkit.community.detectCommunities(graph,
                                                        algo=networkit.community.PLM(graph, refine=True, gamma=gamma))
        cluster.compact()
        cluster_vector = np.array(cluster.getVector())
        cluster_ids = list(cluster.getSubsetIds())
        n_cluster = len(cluster_ids)

    if kind == 'leiden':
        cluster = networkit.community.detectCommunities(graph,
                                                        algo=networkit.community.ParallelLeiden(graph, gamma=gamma))
        cluster.compact()
        cluster_vector = np.array(cluster.getVector())
        cluster_ids = list(cluster.getSubsetIds())
        n_cluster = len(cluster_ids)

    if kind == 'spectral':
        sc = SpectralClustering(n_clusters=n_cluster).fit(affinity_mat)
        cluster_vector = sc.labels_

    return cluster_vector, n_cluster


def kNN_selection(dm, k_percent):
    t0 = time.time()
    knn_dm = copy.deepcopy(dm)
    n, _ = dm.shape
    k = int(n * k_percent)
    top_ixs = np.argpartition(dm, k)[:, :k]
    rows = np.arange(n)[:, None]
    knn_dm[rows, top_ixs] = 0
    print('kNN_selection for k_percent = {} took {:.2f}s'.format(k_percent, time.time()-t0))
    return knn_dm


def get_patient_indices(patient_id, df):
    return [ix for ix, header in enumerate(df.index) if patient_id == header[0]]


def get_feature_from_cluster(cluster_vector, df, kind='absolute'):
    if kind not in ['absolute', 'relative', 'freq']:
        raise ValueError('\'kind\' has to be either \'absolute\', \'relative\' or \'freq\'')

    n_cluster = len(np.unique(cluster_vector))

    patient_ids = np.unique([x[0] for x in df.index])
    n = len(patient_ids)

    feature_vector = np.zeros((n, n_cluster))
    sequences_per_cluster = [[] for _ in range(n_cluster)]

    for patient_ix, tag in enumerate(patient_ids):  # for every patient

        ixs = [ix for ix, header in enumerate(df.index) if tag == header[0]]

        patient_sequences = np.array(df.iloc[ixs]['sequence'].to_list())
        patient_frequencies = np.array(df.iloc[ixs]['freq'].to_list()).astype(float)
        patient_cluster = cluster_vector[ixs]
        # patient_counts = np.array(df.iloc[ixs]['count'].to_list())
        # patient_v = np.array(df.iloc[ixs]['v'].to_list())

        for sequence, cluster, frequency in zip(patient_sequences, patient_cluster, patient_frequencies):
            if kind == 'relative' or kind == 'absolute':
                feature_vector[patient_ix, cluster] += 1
            if kind == 'freq':
                feature_vector[patient_ix, cluster] += frequency

            sequences_per_cluster[cluster].append(sequence)

    if kind == 'relative':
        feature_vector = feature_vector / feature_vector.sum(axis=0)

    return feature_vector, sequences_per_cluster


def get_kNN(train_dm, test_dm, k):
    # TODO
    tree = KDTree(train_dm)
    dd, ii = tree.query(test_dm, k=k)

    return dd, ii


def get_test_cluster_profile(indices, train_cluster_vector, test_df, kind='absolute'):
    n_cluster = len(np.unique(train_cluster_vector))

    patient_ids = np.unique([x[0] for x in test_df.index])
    n = len(patient_ids)

    test_cluster_profile = np.zeros((n, n_cluster))
    sequences_per_cluster = [[] for _ in range(n_cluster)]

    for patient_ix, tag in enumerate(patient_ids):
        ixs = get_patient_indices(tag, test_df)

        patient_sequences = np.array(test_df.iloc[ixs]['sequence'].to_list())
        patient_frequencies = np.array(test_df.iloc[ixs]['freq'].to_list()).astype(float)

        for knn_ixs, sequence, frequency in zip(indices[ixs], patient_sequences, patient_frequencies):
            cluster = mode(train_cluster_vector[knn_ixs])
            if kind == 'relative' or kind == 'absolute':
                test_cluster_profile[patient_ix, cluster] += 1
            if kind == 'freq':
                test_cluster_profile[patient_ix, cluster] += frequency

            sequences_per_cluster[cluster].append(sequence)
    if kind == 'relative':
        test_cluster_profile = test_cluster_profile / test_cluster_profile.sum(axis=0)

    return test_cluster_profile, sequences_per_cluster


def plot_umap(embedding, cluster_vector, umap_title):
    cmap = cm.get_cmap('Set1', max(cluster_vector) + 1)
    x = embedding[:, 0]
    y = embedding[:, 1]
    plt.scatter(x, y, cmap=cmap, c=list(cluster_vector), s=3, alpha=0.5)
    plt.title(umap_title, fontsize=15)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.show()


def plot_similarity_histogram(dm):
    flat_dm = [entry for row in dm for entry in row]
    plt.hist(flat_dm, bins='auto')


def make_classification(train_feature, test_feature, train_response, test_response):
    model = LogisticRegression()
    model.fit(train_feature, train_response)
    y_pred = model.predict(test_feature)
    print(classification_report(test_response, y_pred))


def main():
    df = get_fasta_info()
    dm = get_dm(path=b62, full=True)
    train_df, train_dm, y_train, test_df, test_dm, y_test = get_dm_train_test(df, dm)

    embedding = get_embedding(train_dm)

    train_g = get_graph(train_dm)

    train_cluster_vector, n_cluster = get_cluster(graph=train_g, gamma=1.1, kind='louvain')

    plot_umap(embedding, train_cluster_vector, 'UMAP')

    train_feature_vector, train_sequences_per_cluster = get_feature_from_cluster(train_cluster_vector, train_df)

    distances, indices = get_kNN(train_dm, test_dm, k=10)

    test_feature_vector, test_sequences_per_cluster = get_test_cluster_profile(indices, train_cluster_vector, test_df)

    make_classification(train_feature_vector, test_feature_vector, y_train, y_test)


def kNN_main():

    df = get_fasta_info()
    gamma = 1.11

    dm_paths = [(pam70, 'PAM70'), (b45, 'BLOSUM45'), (b62, 'BLOSUM62')]
    for dm_path, sm_name in dm_paths:
        gt_dm = get_dm(path=dm_path, full=True)
        train_df, gt_train_dm, y_train, test_df, test_dm, y_test = get_dm_train_test(df, gt_dm)
        gt_embedding = get_embedding(gt_train_dm)

        gt_g = get_graph(gt_train_dm)
        gt_cluster_vector, gt_n_cluster = get_cluster(graph=gt_g, gamma=gamma, kind='louvain')
        plot_umap(gt_embedding, gt_cluster_vector, '')

        adjusted_rand_scores, n_clusters = [], []

        k_percents = np.linspace(0, 1, 21)[1:-1]
        for k_percent in k_percents:
            kNN_dm = kNN_selection(gt_train_dm, k_percent)

            kNN_embedding = get_embedding(kNN_dm)
            kNN_g = get_graph(kNN_dm)
            kNN_cluster_vector, kNN_n_cluster = get_cluster(graph=kNN_g, gamma=gamma, kind='louvain')

            plot_umap(kNN_embedding, kNN_cluster_vector, 'kNN embedding with lowest {}% pruned'.format(k_percent*100))
            plot_umap(gt_embedding, kNN_cluster_vector, 'kNN clusters in gt_embedding')

            adjusted_rand_scores.append(adjusted_rand_score(gt_cluster_vector, kNN_cluster_vector))
            n_clusters.append(kNN_n_cluster)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25, 15))

        ax[0].plot(k_percents, n_clusters)
        ax[0].set_title('{}: n_clusters vs k_percents'.format(sm_name))
        ax[0].set_xlabel('lowest k% pruned')
        ax[0].set_ylabel('n_clusters')

        ax[1].plot(k_percents, adjusted_rand_scores)
        ax[1].set_title('{}: ARI vs k_percents'.format(sm_name))
        ax[1].set_xlabel('lowest k% pruned')
        ax[1].set_ylabel('ARI')

        plt.show()


if __name__ == '__main__':
    os.environ['NUMEXPR_MAX_THREADS'] = '52'
    numexpr.set_num_threads(52)

    kNN_main()

