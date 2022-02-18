import copy
import time
import os
import numpy as np
import pandas as pd
import joblib
import networkit
import umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import umap.plot
from Bio import SeqIO
from statistics import mode

from sklearn.model_selection import StratifiedShuffleSplit
from scipy.spatial import KDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

os.environ['NUMEXPR_MAX_THREADS'] = '52'

path_to_dm = 'D:/enno/2022/hiwi/data/dm/pam/BLFUHD_PAM70_10_0.5_DM'
path_to_data = '/home/ubuntu/Enno/gammaDelta/patient_data/all/'
path_to_fasta = '/home/ubuntu/Enno/gammaDelta/patient_data/blfuhd.fasta'


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


def get_dm(name='dummy_dm', full=False):
    dm = joblib.load(name)
    if full:
        dm = (dm + dm.T)

    return dm


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


def get_fasta_info(file=path_to_fasta):
    fasta = list(SeqIO.parse(file, 'fasta'))
    index = [(p, seq_no) for [p, seq_no, f, c, v], seq in [(record.id.split('_'), str(record.seq)) for record in fasta]]
    data = [(seq, f, c, v) for [p, n_seq, f, c, v], seq in [(record.id.split('_'), str(record.seq)) for record in fasta]]
    multi_index = pd.MultiIndex.from_tuples(index, names=['patient', 'seq_no'])

    df = pd.DataFrame(data, index=multi_index, columns=['sequence', 'freq', 'count', 'v'])

    return df


def get_graph(dm):
    timer = time.time()
    m, _ = dm.shape
    g = networkit.Graph(m, weighted=True)
    mask_x, mask_y = np.mask_indices(m, np.tril, -1)
    masking_zip = zip(mask_x, mask_y, dm[mask_x, mask_y])

    for nodeA, nodeB, weight in masking_zip:
        g.addEdge(nodeA, nodeB, weight)

    print('This took %.3f s' % (time.time()-timer))

    return g


def get_embedding(dm):
    reducer = umap.UMAP().fit_transform(dm)

    return reducer


def get_cluster(graph, gamma):
    cluster = networkit.community.detectCommunities(graph,
                                                    algo=networkit.community.PLM(graph, refine=True, gamma=gamma))
    cluster.compact()
    cluster_vector = np.array(cluster.getVector())
    clusters_ids = cluster.getSubsetIds()
    print(clusters_ids)
    print(list(clusters_ids))

    return cluster_vector, clusters_ids


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


def plot_umap(embedding, cluster_vector):
    cmap = cm.get_cmap('Set1', max(cluster_vector) + 1)
    x = embedding[:, 0]
    y = embedding[:, 1]
    plt.scatter(x, y, cmap=cmap, c=list(cluster_vector), s=3, alpha=0.5)
    # plt.title('', fontsize=15)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.show()


def plot_distance_histogram(dm):
    flat_dm = [entry for row in dm for entry in row]
    plt.hist(flat_dm, bins='auto')


def make_classification(train_feature, test_feature, train_response, test_response):
    model = LogisticRegression()
    model.fit(train_feature, train_response)
    y_pred = model.predict(test_feature)
    print(classification_report(test_response, y_pred))


def main():
    df = get_fasta_info()
    dm = get_dm(name=pam70, full=True)
    train_df, train_dm, y_train, test_df, test_dm, y_test = get_dm_train_test(df, dm)

    embedding = get_embedding(train_dm)

    train_g = get_graph(train_dm)

    train_cluster_vector, n_cluster = get_cluster(graph=train_g, gamma=1.0)

    plot_umap(embedding, train_cluster_vector)

    train_feature_vector, train_sequences_per_cluster = get_feature_from_cluster(train_cluster_vector, train_df)

    distances, indices = get_kNN(train_dm, test_dm, k=10)

    test_feature_vector, test_sequences_per_cluster = get_test_cluster_profile(indices, train_cluster_vector, test_df)

    make_classification(train_feature_vector, test_feature_vector, y_train, y_test)


if __name__ == '__main__':
    main()

