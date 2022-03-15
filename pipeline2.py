import copy
import time
import os
import joblib
import numexpr
from collections import Counter
import numpy as np
import pandas as pd
from statistics import mode

from Bio import SeqIO, AlignIO
from Bio.Align import AlignInfo

import networkit
from networkit.community import ParallelLeiden, PLM

import umap
import umap.plot

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import matplotlib

matplotlib.use('Agg')

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

            for ix, (seq, freq, count, trd_v) in enumerate(zip(df['cdr3aa'], df['freq'], df['count'], df['v'])):
                header = '>{}-{}-_sequence-{}_{}_{}_{}'.format(prefix, pat_no, ix + 1, freq, count, trd_v)
                dummy_file.write(header + '\n')
                dummy_file.write(seq + '\n')


def get_am(path_to_am='dummy_dm', full=False):
    am = joblib.load(path_to_am)
    if full:
        am = (am + am.T)

    return am


def shift_similarities_to_zero(am):
    minimum = am.min()
    am = am - minimum
    np.fill_diagonal(am, 0)
    return am


def f(x):
    return 1/(x+1)


def similarities_to_distances(am):
    am = shift_similarities_to_zero(am)
    f_vec = np.vectorize(f)
    distance_matrix = f_vec(am)
    np.fill_diagonal(distance_matrix, 0)

    return distance_matrix


def get_matrix_train_test(df, mat, n_splits=5, test_size=0.2):
    splits = []

    patient_ids = np.unique([x[0] for x in df.index])
    response = np.array([1 if 'BL' in patient or 'FU' in patient else 0 for patient in patient_ids])

    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)

    for train_index, test_index in sss.split(patient_ids, response):
        print('train:', train_index)
        print('test:', test_index)
        train_index.sort()
        test_index.sort()

        X_train, X_test = patient_ids[train_index], patient_ids[test_index]
        y_train, y_test = response[train_index], response[test_index]

        train_ixs = [ix for patient_ixs in [get_patient_indices(tag, df) for tag in X_train] for ix in patient_ixs]
        test_ixs = [ix for patient_ixs in [get_patient_indices(tag, df) for tag in X_test] for ix in patient_ixs]

        train_mat = np.take(np.take(mat, train_ixs, axis=0), train_ixs, axis=1)
        test_mat = np.take(np.take(mat, test_ixs, axis=0), train_ixs, axis=1)

        train_df = df.iloc[train_ixs]
        test_df = df.iloc[test_ixs]

        splits.append((train_df, train_mat, y_train, test_df, test_mat, y_test))

    return splits, train_index, test_index


path_to_fasta = '/home/ubuntu/Enno/gammaDelta/patient_data/blfuhd.fasta'


def get_fasta_info(file=path_to_fasta):
    fasta = list(SeqIO.parse(file, 'fasta'))
    index = [(patient, seq_no) for [patient, seq_no, _, _, _], seq in [(record.id.split('_'), str(record.seq)) for record in fasta]]
    data = [(seq, freq, count, trd_v) for [_, _, freq, count, trd_v], seq in [(record.id.split('_'), str(record.seq)) for record in fasta]]
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


def get_embedding(data):
    reducer = umap.UMAP(metric='precomputed').fit_transform(data)

    return reducer


def eigengap_heuristic(am, plot, loc):
    n, _ = am.shape
    identity = np.identity(n)

    degrees = am.sum(axis=1)
    D = np.zeros((n, n))
    np.fill_diagonal(D, degrees)

    L = identity - np.dot(np.linalg.inv(D), am)
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    if plot:
        plt.title('Largest eigen values of input matrix')
        plt.scatter(np.arange(len(eigenvalues)), eigenvalues, s=1)
        plt.savefig(loc)
        # plt.show()
        plt.clf()

    index_largest_gap = np.argmax(np.diff(eigenvalues))
    n_clusters = index_largest_gap + 1

    return eigenvalues, eigenvectors, n_clusters


def get_cluster(graph=None, gamma=1.0, n_cluster=4, affinity_mat=np.array([]), kind='louvain'):
    cluster_vector = []

    if kind not in ['louvain', 'leiden', 'spectral']:
        raise ValueError('\'kind\' has to be either \'louvain\', \'leiden\' or \'spectral\'')

    if kind == 'louvain':
        if graph is None:
            raise ValueError('\'graph\' has to be a networkit.Graph object.')

        cluster = networkit.community.detectCommunities(graph,
                                                        algo=PLM(graph, refine=True, gamma=gamma))
        cluster.compact()
        cluster_vector = np.array(cluster.getVector())
        cluster_ids = list(cluster.getSubsetIds())
        n_cluster = len(cluster_ids)

    if kind == 'leiden':
        if graph is None:
            raise ValueError('\'graph\' has to be a networkit.Graph object.')

        cluster = networkit.community.detectCommunities(graph,
                                                        algo=ParallelLeiden(graph, gamma=gamma))
        cluster.compact()
        cluster_vector = np.array(cluster.getVector())
        cluster_ids = list(cluster.getSubsetIds())
        n_cluster = len(cluster_ids)

    if kind == 'spectral':
        if not affinity_mat.size > 0:
            raise ValueError('\'affinity_mat\' has to be a symmetric matrix with size > 0.')

        sc = SpectralClustering(n_clusters=n_cluster, affinity='precomputed').fit(affinity_mat)
        cluster_vector = sc.labels_

    return cluster_vector, n_cluster


def kNN_selection(mat, k_percent, kind='affinity'):
    if kind not in ['affinity', 'distance']:
        raise ValueError('\'kind\' has to be either \'affinity\' or \'distance\'.')
    t0 = time.time()
    knn_mat = copy.deepcopy(mat)
    n, _ = mat.shape
    k = int(n * k_percent)
    top_ixs = []

    if kind == 'affinity':
        top_ixs = np.argpartition(mat, k)[:, :k]
    if kind == 'distance':
        top_ixs = np.argpartition(mat, -k)[:, -k:]

    rows = np.arange(n)[:, None]
    knn_mat[rows, top_ixs] = 0
    print('kNN_selection (kind={}) for k_percent = {} took {:.2f}s'.format(kind, k_percent, time.time()-t0))
    return knn_mat


def get_patient_indices(patient_id, df):
    return [ix for ix, header in enumerate(df.index) if patient_id == header[0]]


def get_cohorte_indices(cohorte: str, df):
    patient_ids = np.unique([_[0] for _ in df.index])
    cohorte_patient_ids = [_ for _ in patient_ids if cohorte in _]
    cohorte_indices = [get_patient_indices(patient_id, df) for patient_id in cohorte_patient_ids]
    cohorte_indices = [ix for patient_indices in cohorte_indices for ix in patient_indices]
    cohorte_indices.sort()

    return cohorte_indices


def exclude_class(class_label: str, df, A):
    del_ixs = get_cohorte_indices(class_label, df)
    reduced_A = np.delete(A, del_ixs, axis=0)
    reduced_A = np.delete(reduced_A, del_ixs, axis=1)
    reduced_df = df.drop(df.index[del_ixs])

    return reduced_A, reduced_df


def get_train_F(cluster_vector, df, kind='absolute'):
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
        patient_Vs = np.array(df.iloc[ixs]['v'].to_list())
        patient_frequencies = np.array(df.iloc[ixs]['freq'].to_list()).astype(float)
        patient_cluster = cluster_vector[ixs]

        for sequence, cluster, frequency, v in zip(patient_sequences, patient_cluster, patient_frequencies, patient_Vs):
            if kind == 'relative' or kind == 'absolute':
                feature_vector[patient_ix, cluster] += 1
            if kind == 'freq' or kind == 'relative_freq':
                feature_vector[patient_ix, cluster] += frequency

            sequences_per_cluster[cluster].append([tag[:2], tag[:-1], sequence, v])

    if kind == 'relative':
        feature_vector = feature_vector / feature_vector.sum(axis=0)

    if kind == 'relative_freq':
        feature_vector = feature_vector / feature_vector.sum(axis=0)

    return feature_vector, sequences_per_cluster


def get_test_C(train_D, train_C, test_D, n_neighbors=111):
    knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_clf.fit(train_D, train_C)

    test_C = knn_clf.predict(test_D)

    return test_C


def get_test_F(test_C, test_df, n_cluster, kind='relative'):
    patient_ids = np.unique([x[0] for x in test_df.index])
    n = len(patient_ids)
    test_F = np.zeros((n, n_cluster))
    test_SPC = [[] for _ in range(n_cluster)]

    for patient_ix, tag in enumerate(patient_ids):  # for every patient
        ixs = [ix for ix, header in enumerate(test_df.index) if tag == header[0]]

        patient_sequences = np.array(test_df.iloc[ixs]['sequence'].to_list())
        patient_Vs = np.array(test_df.iloc[ixs]['v'].to_list())
        patient_frequencies = np.array(test_df.iloc[ixs]['freq'].to_list()).astype(float)
        patient_cluster = test_C[ixs]

        for sequence, cluster, frequency, v in zip(patient_sequences, patient_cluster, patient_frequencies, patient_Vs):

            if kind == 'relative' or kind == 'absolute':
                test_F[patient_ix, cluster] += 1

            test_SPC[cluster].append([tag[:2], tag[:-1], sequence, v])

    if kind == 'relative':
        test_F = test_F / test_F.sum(axis=0)

    np.nan_to_num(test_F, copy=False)

    return test_F, test_SPC


def get_kNN(train_dm, test_dm, k=11):
    nn = NearestNeighbors(n_neighbors=k, metric='precomputed', algorithm='auto')
    nn.fit(train_dm)
    dd, ii = nn.kneighbors(test_dm, n_neighbors=11)
    return dd, ii


def get_test_cluster_profile(indices, train_cluster_vector, test_df, kind='relative'):
    # TODO adjust output so it can be visualized
    n_cluster = len(np.unique(train_cluster_vector))

    patient_ids = np.unique([x[0] for x in test_df.index])
    n = len(patient_ids)

    test_cluster_profile = np.zeros((n, n_cluster))
    sequences_per_cluster = [[] for _ in range(n_cluster)]

    for patient_ix, tag in enumerate(patient_ids):
        ixs = get_patient_indices(tag, test_df)

        patient_sequences = np.array(test_df.iloc[ixs]['sequence'].to_list())
        patient_frequencies = np.array(test_df.iloc[ixs]['freq'].to_list()).astype(float)
        patient_Vs = np.array(test_df.iloc[ixs]['v'].to_list())

        for knn_ixs, sequence, frequency, v in zip(indices[ixs], patient_sequences, patient_frequencies, patient_Vs):
            cluster = mode(train_cluster_vector[knn_ixs])
            if kind == 'relative' or kind == 'absolute':
                test_cluster_profile[patient_ix, cluster] += 1
            if kind == 'freq':
                test_cluster_profile[patient_ix, cluster] += frequency

            sequences_per_cluster[cluster].append([tag[:2], tag[:-1], sequence, v])
    if kind == 'relative':
        test_cluster_profile = test_cluster_profile / test_cluster_profile.sum(axis=0)

    return test_cluster_profile, sequences_per_cluster


def plot_umap(embedding, cluster_vector, umap_title, loc):
    cmap = cm.get_cmap('Set1', max(cluster_vector) + 1)
    x = embedding[:, 0]
    y = embedding[:, 1]
    plt.scatter(x, y, cmap=cmap, c=list(cluster_vector), s=3, alpha=0.5)
    plt.title(umap_title, fontsize=15)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.savefig(loc)
    # plt.show()
    plt.clf()


def plot_similarity_histogram(dm):
    flat_dm = [entry for row in dm for entry in row]
    plt.hist(flat_dm, bins='auto')


def make_classification(train_feature, test_feature, train_response, test_response):
    model = LogisticRegression()
    model.fit(train_feature, train_response)
    y_pred = model.predict(test_feature)
    print(classification_report(test_response, y_pred))

    f1 = f1_score(y_true=test_response, y_pred=y_pred)
    bal_acc = balanced_accuracy_score(y_true=test_response, y_pred=y_pred)
    prec = precision_score(y_true=test_response, y_pred=y_pred)
    sens = recall_score(y_true=test_response, y_pred=y_pred, pos_label=1)
    spec = recall_score(y_true=test_response, y_pred=y_pred, pos_label=0)

    return [f1, bal_acc, prec, sens, spec]


def visualize_cluster_distributions(cluster_info, sm, ck):

    classes = ['HD', 'BL', 'FU']

    for cix, cluster in enumerate(cluster_info):
        temp_df = pd.DataFrame(columns=['Class', 'ID', 'seq', 'v'])

        for ix, (cohorte, tag, seq, v) in enumerate(cluster):
            temp_df.loc[ix] = [cohorte, tag, seq, v]

        sub_1 = temp_df.Class.value_counts()
        sub_2 = temp_df.v.value_counts()

        visual_df = pd.concat([sub_1, sub_2], axis=1, keys=['classes', 'TRDV'])

        axes = visual_df.plot(kind='pie', subplots=True, figsize=(10, 6))
        for ax in axes:
            ax.set_aspect('equal')
            ax.yaxis.set_label_coords(-0.15, 0.5)
            ax.legend(bbox_to_anchor=(2.1, 0.5), loc='center right')
        plt.subplots_adjust(wspace=1.4)
        plt.savefig('/home/ubuntu/Enno/gammaDelta/{}/{}_{}_{}.png'.format(sm, sm, ck, cix), bbox_inches='tight')


def get_consensus_sequences(cluster_info, percent_occurence, sm, ck):
    for cix, cluster in enumerate(cluster_info):
        temp_df = pd.DataFrame(columns=['Class', 'ID', 'seq', 'v'])

        for ix, (cohorte, tag, seq, v) in enumerate(cluster):
            temp_df.loc[ix] = [cohorte, tag, seq, v]

        list_seq = temp_df.seq.to_list()
        list_name = temp_df.ID.to_list()
        temp_file = open("temp_fasta.fasta", "w")
        for i in range(len(list_seq)):
            temp_file.write(">" + list_name[i] + "\n" + list_seq[i] + "\n")

        alignment = AlignIO.read(temp_file, 'fasta')
        summary_align = AlignInfo.SummaryInfo(alignment)
        summary_align.dumb_consensus(percent_occurence)
        temp_file.close()


def classification_main(sm_name,
                        dm_path,
                        n_splits, test_size,
                        class_label_to_exclude=None,
                        k_=None, g_=None,
                        knn_clf_k=11,
                        plot_eigengap=False, plot_full_UMAP=False, plot_excluded_class_UMAP=False, plot_UMAP_w_clusters=False):

    print('Starting run on {}'.format(sm_name))

    overall_results = []

    if k_ is None:
        k_ = [0]

    train_graph = None
    n_sc_cluster = 0

    df = get_fasta_info()
    full_A = get_am(dm_path, full=True)
    full_A = shift_similarities_to_zero(full_A)           # shifted affinity matrix A
    full_D = similarities_to_distances(full_A)

    print('Data acquired, shifted, and transformed.')

    full_embedding = get_embedding(full_D)

    if plot_full_UMAP:
        n, _ = full_D.shape
        plot_umap(full_embedding, [1]*n, '{} - all sequences included'.format(sm_name),
                  loc='{}_full_umap.png'.format(sm_name))

    if class_label_to_exclude:
        A, reduced_df = exclude_class(class_label_to_exclude, df, full_A)
        D, _ = exclude_class(class_label_to_exclude, df, full_D)

        print('Exclusion accomplished.')

        if plot_excluded_class_UMAP:
            embedding = get_embedding(D)
            n, _ = D.shape
            plot_umap(embedding, [1] * n, '{} - {} excluded'.format(sm_name, class_label_to_exclude),
                      loc='/home/ubuntu/Enno/gammaDelta/{}/{}_{}_excluded_umap.png'.format(sm_name, sm_name, class_label_to_exclude))
    else:
        A = full_A
        reduced_df = df

    print('Now starting k% iteration.')

    for k_percent in k_:
        if k_percent == 0:
            A_in_use = A
        else:
            A_in_use = kNN_selection(A, k_percent)

        cv_splits = get_matrix_train_test(df=reduced_df, mat=A_in_use, n_splits=n_splits, test_size=test_size)
        cv_performance, cv_self_performance = [], []

        for cluster_kind in ['spectral', 'leiden']:
            # for feature_kind in ['absolute', 'relative', 'freq', 'relative_freq']:
            # print('Now starting {} clustering.'.format(cluster_kind))

            first_loop = True
            for train_df, train_A, y_train, test_df, test_A, y_test in cv_splits:
                print(len(train_df), len(test_df))
                print(train_A.shape, test_A.shape)
                print(y_train, y_test)

                if cluster_kind in ['louvain', 'leiden']:
                    train_graph = get_graph(train_A)

                elif cluster_kind == 'spectral':
                    eigenvalues, eigenvectors, n_sc_cluster = eigengap_heuristic(train_A, plot=plot_eigengap,
                                                                                 loc='/home/ubuntu/Enno/gammaDelta/{}/{}_spectral_eigengap.png'.format(sm_name, sm_name))
                    if n_sc_cluster > 100:
                        n_sc_cluster = 50

                train_cluster_vector, n_cluster = get_cluster(graph=train_graph, gamma=g_,
                                                              affinity_mat=train_A,
                                                              n_cluster=n_sc_cluster,
                                                              kind=cluster_kind)

                if plot_UMAP_w_clusters:
                    train_D = similarities_to_distances(train_A)
                    train_mbed = get_embedding(train_D)
                    plot_umap(train_mbed, train_cluster_vector, '{} - {} excluded'.format(sm_name, class_label_to_exclude),
                              loc='/home/ubuntu/Enno/gammaDelta/{}/{}_train_UMAP_w_{}.png'.format(sm_name, sm_name, cluster_kind))

                # TODO kind iterator
                train_feature_vector, train_sequences_per_cluster = get_train_F(train_cluster_vector, train_df, kind='relative')
                distances, indices = get_kNN(train_A, test_A, k=knn_clf_k)
                test_feature_vector, test_sequences_per_cluster = get_test_cluster_profile(indices, train_cluster_vector,
                                                                                           test_df)

                if first_loop:
                    visualize_cluster_distributions(train_sequences_per_cluster, sm=sm_name, ck=cluster_kind)
                    first_loop = False

                performance = make_classification(train_feature_vector, test_feature_vector, y_train, y_test)
                print(performance)

                self_performance = make_classification(train_feature_vector, train_feature_vector, y_train, y_train)
                print(self_performance)

                cv_performance.append(performance)
                cv_self_performance.append(self_performance)

            average_cv_performance = np.average(cv_performance, axis=0)
            average_cv_self_performance = np.average(cv_self_performance, axis=0)

            cv_performance, cv_self_performance = [], []

            overall_results.append([sm_name, cluster_kind, n_cluster, k_percent,
                                    average_cv_performance[0],
                                    average_cv_performance[1],
                                    average_cv_performance[2],
                                    average_cv_performance[3],
                                    average_cv_performance[4]])

            overall_results.append([sm_name+' self', cluster_kind, n_cluster, k_percent,
                                    average_cv_self_performance[0],
                                    average_cv_self_performance[1],
                                    average_cv_self_performance[2],
                                    average_cv_self_performance[3],
                                    average_cv_self_performance[4]])

    result_df = pd.DataFrame(columns=['sm', 'cluster_kind', 'n_cluster', 'k_percent', 'f1', 'bal_acc', 'prec', 'sens', 'spec'])

    for ix, result in enumerate(overall_results):
        result_df.loc[ix] = result

    result_df.to_csv('/home/ubuntu/Enno/gammaDelta/{}/{}_result.csv'.format(sm_name, sm_name))
    print('{} run done.'.format(sm_name))


def main():
    df = get_fasta_info()
    dm = get_am(path_to_am=b62, full=True)
    split, train_I, test_I = get_matrix_train_test(df, dm)
    train_df, train_dm, y_train, test_df, test_dm, y_test = split[0]
    embedding = get_embedding(train_dm)

    train_G = get_graph(train_dm)

    train_C, n_cluster = get_cluster(graph=train_G, gamma=1.1, kind='louvain')

    plot_umap(embedding, train_C, 'UMAP', '')

    train_feature_vector, train_sequences_per_cluster = get_train_F(train_C, train_df)

    distances, indices = get_kNN(train_dm, test_dm, k=10)

    test_feature_vector, test_sequences_per_cluster = get_test_cluster_profile(indices, train_C, test_df)

    make_classification(train_feature_vector, test_feature_vector, y_train, y_test)


def kNN_main():

    df = get_fasta_info()
    gamma = 1.11

    dm_paths = [(pam70, 'PAM70'), (b45, 'BLOSUM45'), (b62, 'BLOSUM62')]
    for dm_path, sm_name in dm_paths:
        gt_dm = get_am(path_to_am=dm_path, full=True)
        train_df, gt_train_dm, y_train, test_df, test_dm, y_test = get_matrix_train_test(df, gt_dm)
        gt_embedding = get_embedding(gt_train_dm)

        gt_g = get_graph(gt_train_dm)
        gt_cluster_vector, gt_n_cluster = get_cluster(graph=gt_g, gamma=gamma, kind='louvain')
        plot_umap(gt_embedding, gt_cluster_vector, '', '')

        adjusted_rand_scores, n_clusters = [], []

        k_percents = np.linspace(0, 1, 21)[1:-1]
        for k_percent in k_percents:
            kNN_dm = kNN_selection(gt_train_dm, k_percent)

            kNN_embedding = get_embedding(kNN_dm)
            kNN_g = get_graph(kNN_dm)
            kNN_cluster_vector, kNN_n_cluster = get_cluster(graph=kNN_g, gamma=gamma, kind='louvain')

            plot_umap(kNN_embedding, kNN_cluster_vector, 'kNN embedding with lowest {}% pruned'.format(k_percent*100), '')
            plot_umap(gt_embedding, kNN_cluster_vector, 'kNN clusters in gt_embedding', '')

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


def spectral_n_main(sm_path, sm_name, kind):
    performance_df = pd.DataFrame(columns=['NC', 'BA', 'F1', 'PR', 'SP', 'SN'])
    self_performance_df = pd.DataFrame(columns=['NC', 'BA', 'F1', 'PR', 'SP', 'SN'])

    full_df = get_fasta_info()
    full_A = get_am(sm_path, full=True)
    full_A = shift_similarities_to_zero(full_A)  # shifted affinity matrix A
    full_D = similarities_to_distances(full_A)

    red_A, reduced_df = exclude_class('FU', full_df, full_A)
    red_D, _ = exclude_class('FU', full_df, full_D)

    cv_splits, train_I, test_I = get_matrix_train_test(df=reduced_df, mat=red_A, n_splits=1, test_size=0.2)
    train_df, train_A, train_Y, test_df, test_A, test_Y = cv_splits[0]

    test_D = similarities_to_distances(test_A)
    train_D = similarities_to_distances(train_A)

    # red_mbed = get_embedding(red_D)
    # train_mbed = get_embedding(train_D)

    for ix, n_scluster in enumerate([3, 6, 9, 12, 16, 24, 32, 48, 54, 66, 90, 120, 240, 320, 480, 600, 700, 800, 1000,  1250,  1500]):
        print(ix, n_scluster)
        train_C, n_cluster = get_cluster(graph=None, gamma=1,
                                                      affinity_mat=train_A,
                                                      n_cluster=n_scluster,
                                                      kind='spectral')

        train_F, train_SPC = get_train_F(train_C, train_df, kind=kind)

        test_C = get_test_C(train_D, train_C, test_D)
        test_F, test_SPC = get_test_F(test_C, test_df, n_cluster, kind=kind)

        model = LogisticRegression(class_weight='balanced')
        model.fit(train_F, train_Y)

        pred_Y = model.predict(test_F)
        performance_df.loc[ix] = [n_cluster,
                                  balanced_accuracy_score(test_Y, pred_Y),
                                  f1_score(test_Y, pred_Y),
                                  precision_score(test_Y, pred_Y),
                                  recall_score(test_Y, pred_Y),
                                  recall_score(test_Y, pred_Y, pos_label=0)]

        pred_Y = model.predict(train_F)
        self_performance_df.loc[ix] = [n_cluster,
                                       balanced_accuracy_score(train_Y, pred_Y),
                                       f1_score(train_Y, pred_Y),
                                       precision_score(train_Y, pred_Y),
                                       recall_score(train_Y, pred_Y),
                                       recall_score(train_Y, pred_Y, pos_label=0)]

    performance_df[['BA', 'F1', 'PR', 'SP', 'SN']].plot(figsize=(16, 8))
    plt.ylabel('classification performance')
    plt.xlabel('number of clusters')
    plt.title('{} classifier performance on test data using different spectral n'.format(sm_name))
    plt.xticks(ticks=performance_df.index, labels=performance_df['NC'].astype(int))
    plt.savefig('{}/{}_test_performance_spectral_n.png'.format(sm_name, sm_name))

    self_performance_df[['BA', 'F1', 'PR', 'SP', 'SN']].plot(figsize=(16, 8))
    plt.ylabel('classification performance')
    plt.xlabel('number of clusters')
    plt.title('{} classifier performance on train data using spectral n'.format(sm_name))
    plt.xticks(ticks=performance_df.index, labels=performance_df['NC'].astype(int))
    plt.savefig('{}/{}_self_performance_spectral_n.png'.format(sm_name, sm_name))


def gamma_main(sm_path, sm_name, kind):
    performance_df = pd.DataFrame(columns=['NC', 'BA', 'F1', 'PR', 'SP', 'SN'])
    self_performance_df = pd.DataFrame(columns=['NC', 'BA', 'F1', 'PR', 'SP', 'SN'])

    full_df = get_fasta_info()
    full_A = get_am(sm_path, full=True)
    full_A = shift_similarities_to_zero(full_A)  # shifted affinity matrix A
    full_D = similarities_to_distances(full_A)

    red_A, reduced_df = exclude_class('FU', full_df, full_A)
    red_D, _ = exclude_class('FU', full_df, full_D)

    cv_splits, train_I, test_I = get_matrix_train_test(df=reduced_df, mat=red_A, n_splits=1, test_size=0.2)
    train_df, train_A, train_Y, test_df, test_A, test_Y = cv_splits[0]

    train_G = get_graph(train_A)
    train_D = similarities_to_distances(train_A)
    test_D = similarities_to_distances(test_A)

    # red_mbed = get_embedding(red_D)
    # train_mbed = get_embedding(train_D)

    for ix, g in enumerate(
            [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17,
             1.18, 1.19, 1.2]):
        train_C, n_cluster = get_cluster(graph=train_G, gamma=g,
                                                      affinity_mat=np.array([]),
                                                      n_cluster=0,
                                                      kind='leiden')

        train_F, train_SPC = get_train_F(train_C, train_df, kind=kind)

        test_C = get_test_C(train_D, train_C, test_D)
        test_F, test_SPC = get_test_F(test_C, test_df, n_cluster, kind=kind)

        model = LogisticRegression(class_weight='balanced')
        model.fit(train_F, train_Y)

        pred_Y = model.predict(test_F)
        performance_df.loc[ix] = [n_cluster,
                                  balanced_accuracy_score(test_Y, pred_Y),
                                  f1_score(test_Y, pred_Y),
                                  precision_score(test_Y, pred_Y),
                                  recall_score(test_Y, pred_Y),
                                  recall_score(test_Y, pred_Y, pos_label=0)]

        pred_Y = model.predict(train_F)
        self_performance_df.loc[ix] = [n_cluster,
                                       balanced_accuracy_score(train_Y, pred_Y),
                                       f1_score(train_Y, pred_Y),
                                       precision_score(train_Y, pred_Y),
                                       recall_score(train_Y, pred_Y),
                                       recall_score(train_Y, pred_Y, pos_label=0)]

    performance_df[['BA', 'F1', 'PR', 'SP', 'SN']].plot(figsize=(16, 8))
    plt.ylabel('classification performance')
    plt.xlabel('number of clusters')
    plt.title('{} classifier performance on test data using different gammas'.format(sm_name))
    plt.xticks(ticks=performance_df.index, labels=performance_df['NC'].astype(int))
    plt.savefig('{}/{}_test_performance_gammas.png'.format(sm_name, sm_name))

    self_performance_df[['BA', 'F1', 'PR', 'SP', 'SN']].plot(figsize=(16, 8))
    plt.ylabel('classification performance')
    plt.xlabel('number of clusters')
    plt.title('{} classifier performance on train data using different gammas'.format(sm_name))
    plt.xticks(ticks=performance_df.index, labels=performance_df['NC'].astype(int))
    plt.savefig('{}/{}_self_performance_gammas.png'.format(sm_name, sm_name))


if __name__ == '__main__':
    os.environ['NUMEXPR_MAX_THREADS'] = '52'
    numexpr.set_num_threads(52)

    names = [('PAM70', pam70), ('BLOSUM45', b45), ('BLOSUM62', b62)]

    for name, path in names:
        spectral_n_main(path, name)
        gamma_main(path, name)
