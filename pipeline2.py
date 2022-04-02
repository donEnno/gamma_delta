import copy
import time
import os
import joblib
import numexpr
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

# global parameters
matplotlib.use('Agg')
os.environ['NUMEXPR_MAX_THREADS'] = '52'

# global paths
path_to_dm = 'D:/enno/2022/hiwi/data/dm/pam/BLFUHD_PAM70_10_0.5_DM'
path_to_data = '/home/ubuntu/Enno/gammaDelta/patient_data/all/'
path_to_fasta = '/home/ubuntu/Enno/gammaDelta/patient_data/blfuhd.fasta'
dm_root = '/home/ubuntu/Enno/mnt/volume/dm_in_use/'

# substitution matrices
b45 = dm_root + 'BLFUHD_BLOSUM45_1_0.1_DM'
b62 = dm_root + 'BLFUHD_BLOSUM62_10_0.5_DM'
pam70 = dm_root + 'BLFUHD_PAM70_1_0.1_DM'


def write_fasta():
    """
    Reads all sequences from raw .txt files and writes them to .fasta file format with the following header
    format: >COHORTE_#PATIENT_sequence-_#SEQUENCE_SEQ-FREQ_SEQ-COUNT_-SEQ-TRDV.
    This way the headers are easily converted to a dataframe.
    """
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
    """
    :param path_to_am:
    :param full: Set True to obtain full symmetric AM.
    :return: AM
    """
    am = joblib.load(path_to_am)
    if full:
        am = (am + am.T)

    return am


def shift_similarities_to_zero(am):
    """
    :param am:
    :return: Initial AM but values are shifted to zero to avoid negative affinities.
    """
    minimum = am.min()
    am = am - minimum
    np.fill_diagonal(am, 0)
    return am


def f(x):
    """
    Affinity to distance transformation.
    :param x:
    :return: distance
    """
    return 1 / (x + 1)


def similarities_to_distances(am):
    """
    Converts input AM to DM.
    :param am: DM
    :return:
    """
    am = shift_similarities_to_zero(am)
    f_vec = np.vectorize(f)
    distance_matrix = f_vec(am)
    np.fill_diagonal(distance_matrix, 0)

    return distance_matrix


def get_matrix_train_test(df, mat, n_splits=5, test_size=0.2):
    """

    :param df: Dataframe that supports the data.
    :param mat: Matrix to split on.
    :param n_splits:
    :param test_size:
    :return: splits: Splits of the supporting dataframe, matrix, and repsonse.
             train_indices: Indices of patients to train on.
             test_indices: Indices of patients to test on.
    """
    splits = []
    train_indices = []
    test_indices = []

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
        train_indices.append(train_index)
        test_indices.append(test_index)

    return splits, train_indices, test_indices


def get_fasta_info(file=path_to_fasta):
    """
    Reads .fasta file and recovers a dataframe from it.
    :param file:
    :return: dataframe
    """
    fasta = list(SeqIO.parse(file, 'fasta'))
    index = [(patient, seq_no) for [patient, seq_no, _, _, _], seq in
             [(record.id.split('_'), str(record.seq)) for record in fasta]]
    data = [(seq, freq, count, trd_v) for [_, _, freq, count, trd_v], seq in
            [(record.id.split('_'), str(record.seq)) for record in fasta]]
    multi_index = pd.MultiIndex.from_tuples(index, names=['patient', 'seq_no'])

    df = pd.DataFrame(data, index=multi_index, columns=['sequence', 'freq', 'count', 'v'])

    return df


def get_graph(dm):
    """
    Builds a networkit graph from the input.
    :param dm:
    :return: networkit graph
    """
    t0 = time.time()
    m, _ = dm.shape
    g = networkit.Graph(m, weighted=True)
    mask_x, mask_y = np.mask_indices(m, np.tril, -1)
    masking_zip = zip(mask_x, mask_y, dm[mask_x, mask_y])

    for nodeA, nodeB, weight in masking_zip:
        if weight == 0:
            continue
        g.addEdge(nodeA, nodeB, weight)

    print('The graph construction took %.3f s' % (time.time() - t0))

    return g


def get_embedding(data):
    """
    Data is an AM that is porcessed by UMAP.
    :param data:
    :return: UMAP embedding of the data.
    """
    reducer = umap.UMAP(metric='precomputed').fit_transform(data)

    return reducer


def eigengap_heuristic(am, plot, plot_path):
    """
    Computes the eigenap of the AM as stated in
    https://www.tml.cs.uni-tuebingen.de/team/luxburg/publications/Luxburg07_tutorial.pdf.
    :param am:
    :param plot: True if eigengap should be plotted.
    :param plot_path:
    :return: Number of clusters to use for spectral clustering as suggested by the eigengap theory.
    """
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
        plt.savefig(plot_path)
        # plt.show()
        plt.clf()

    index_largest_gap = np.argmax(np.diff(eigenvalues))
    n_clusters = index_largest_gap + 1

    return eigenvalues, eigenvectors, n_clusters


def get_cluster(graph=None, gamma=1.0, n_cluster=4, affinity_mat=np.array([]), kind='louvain'):
    """
    Perform clustering dependend on input params.
    :param graph: obligatory if kind='leiden' or 'louvain'
    :param gamma: obligatory if kind='leiden' or 'louvain'
    :param n_cluster: obligatory if kind='spectral'
    :param affinity_mat: obligatory if kind='leiden' or 'louvain'
    :param kind:
    :return: cluster vector and number of clusters found
    """
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
    """
    Sets upper/lower k_percent of the input matrix to 0 dependent on kind.
    :param mat:
    :param k_percent:
    :param kind:
    :return:
    """
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
    knn_mat[top_ixs, rows] = 0
    print('kNN_selection (kind={}) for k_percent = {} took {:.2f}s'.format(kind, k_percent, time.time() - t0))
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
    """
    Builds training feature vector from the input clustering.
    :param cluster_vector:
    :param df:
    :param kind:
    :return:
    """
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
    """
    Performs kNN classification of the test data to obtain a test cluster vector.
    :param train_D:
    :param train_C:
    :param test_D:
    :param n_neighbors:
    :return:
    """
    knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_clf.fit(train_D, train_C)

    test_C = knn_clf.predict(test_D)

    return test_C


def get_test_F(test_C, test_df, n_cluster, kind='relative'):
    """
    Build test feature vector from the test clustering obtained through kNN classification.
    :param test_C:
    :param test_df:
    :param n_cluster:
    :param kind:
    :return:
    """
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
    """
    kNN classification of the test data.
    :param train_dm:
    :param test_dm:
    :param k:
    :return:
    """
    nn = NearestNeighbors(n_neighbors=k, metric='precomputed', algorithm='auto')
    nn.fit(train_dm)
    dd, ii = nn.kneighbors(test_dm, n_neighbors=11)
    return dd, ii


def make_classification(train_feature, test_feature, train_response, test_response):
    """
    Make classification on a single fold.
    :param train_feature:
    :param test_feature:
    :param train_response:
    :param test_response:
    :return:
    """
    model = LogisticRegression(class_weight='balanced', max_iter=50000)
    model.fit(train_feature, train_response)
    y_pred = model.predict(test_feature)
    print(classification_report(test_response, y_pred))

    f1 = f1_score(y_true=test_response, y_pred=y_pred)
    bal_acc = balanced_accuracy_score(y_true=test_response, y_pred=y_pred)
    prec = precision_score(y_true=test_response, y_pred=y_pred)
    sens = recall_score(y_true=test_response, y_pred=y_pred, pos_label=1)
    spec = recall_score(y_true=test_response, y_pred=y_pred, pos_label=0)

    return [f1, bal_acc, prec, sens, spec]


# The following methods serve the purpose of plotting different parts of the data respectively the data at different
# steps in the workflow.

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
    """
    Get consensus sequnce per cluster dependent on a threshold.
    :param cluster_info:
    :param percent_occurence: threshold on number of sequnces the amino acids have to appear in to be considered part
                              of the consenus sequence.
    :param sm:
    :param ck:
    :return:
    """
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


def kNN_main(sm_path, sm_name):
    """
    Script to run on the virtual machine.
    :param sm_path:
    :param sm_name:
    :return:
    """
    K = [x/10 for x in list(range(3, 10))]
    N_SCLUSTER = np.geomspace(5, 500, 20).astype(int)
    # [3, 6, 9, 12, 16, 24, 32, 48, 54, 66, 90, 120, 240, 320, 480, 600, 700, 800, 1000, 1250, 1500]
    # GAMMAS = [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17,
    #           1.18, 1.19, 1.2]
    # cluster_parameter_list = list(zip(N_SCLUSTER, GAMMAS))

    performance = []
    self_performance = []
    ari = []

    full_df = get_fasta_info()
    full_A = get_am(sm_path, full=True)
    full_A = shift_similarities_to_zero(full_A)  # shifted affinity matrix A
    full_D = similarities_to_distances(full_A)

    full_A, reduced_df = exclude_class('FU', full_df, full_A)
    full_D, _ = exclude_class('FU', full_df, full_D)

    # full_G = get_graph(full_A)

    for n_s_cluster in N_SCLUSTER:  # , gamma
        full_spectral_C, n_full_spectral = get_cluster(graph=None, gamma=1, n_cluster=n_s_cluster, affinity_mat=full_A,
                                                       kind='spectral')
        # full_leiden_C, n_full_leiden = get_cluster(graph=full_G, gamma=gamma, n_cluster=0, affinity_mat=np.array([]),
        #                                            kind='leiden')

        for k_ in K:
            kNN_A = kNN_selection(full_A, k_, kind='affinity')
            kNN_D = shift_similarities_to_zero(kNN_A)
            # kNN_G = get_graph(kNN_A)

            kNN_full_spectral_C, n_kNN_full_spectral = get_cluster(graph=None, gamma=1, n_cluster=n_s_cluster,
                                                                   affinity_mat=kNN_A, kind='spectral')
            # kNN_full_leiden_C, n_kNN_full_leiden = get_cluster(graph=kNN_G, gamma=gamma, n_cluster=0,
            #                                                    affinity_mat=np.array([]), kind='leiden')

            # FOR SPLIT IN SPLITS
            cv_splits, train_I, test_I = get_matrix_train_test(df=reduced_df, mat=kNN_A, n_splits=1, test_size=0.2)
            train_df, kNN_train_A, train_Y, test_df, kNN_test_A, test_Y = cv_splits[0]

            kNN_train_D = similarities_to_distances(kNN_train_A)
            kNN_test_D = similarities_to_distances(kNN_test_A)
            # kNN_train_G = get_graph(kNN_train_A)

            # spectral and leiden cluster vectors
            kNN_train_spectral_C, n_kNN_train_spectral = get_cluster(graph=None, gamma=1, n_cluster=n_s_cluster,
                                                                     affinity_mat=kNN_train_A, kind='spectral')
            # kNN_train_leiden_C, n_kNN_train_leiden = get_cluster(graph=kNN_train_G, gamma=gamma, n_cluster=0,
            #                                                      affinity_mat=np.array([]), kind='leiden')

            # spectral and leiden relative
            kNN_train_spectral_F_rel, _ = get_train_F(kNN_train_spectral_C, train_df, kind='relative')
            # kNN_train_leiden_F_rel, _ = get_train_F(kNN_train_leiden_C, train_df, kind='relative')

            kNN_test_spectral_C_rel = get_test_C(kNN_train_D, kNN_train_spectral_C, kNN_test_D)
            # kNN_test_leiden_C_rel = get_test_C(kNN_train_D, kNN_train_leiden_C, kNN_test_D)

            kNN_test_spectral_F_rel, _ = get_test_F(kNN_test_spectral_C_rel, test_df, n_kNN_train_spectral,
                                                    kind='relative')
            # kNN_test_leiden_F_rel, _ = get_test_F(kNN_test_leiden_C_rel, test_df, n_kNN_train_leiden, kind='relative')

            # spectral and leiden absolute
            kNN_train_spectral_F_abs, _ = get_train_F(kNN_train_spectral_C, train_df, kind='absolute')
            # kNN_train_leiden_F_abs, _ = get_train_F(kNN_train_leiden_C, train_df, kind='absolute')

            kNN_test_spectral_C_abs = get_test_C(kNN_train_D, kNN_train_spectral_C, kNN_test_D)
            # kNN_test_leiden_C_abs = get_test_C(kNN_train_D, kNN_train_leiden_C, kNN_test_D)

            kNN_test_spectral_F_abs, _ = get_test_F(kNN_test_spectral_C_abs, test_df, n_kNN_train_spectral,
                                                    kind='absolute')
            # kNN_test_leiden_F_abs, _ = get_test_F(kNN_test_leiden_C_abs, test_df, n_kNN_train_leiden, kind='absolute')

            # classification
            model = LogisticRegression(class_weight='balanced', max_iter=50000)

            # spectral relative and absolute
            model.fit(kNN_train_spectral_F_abs, train_Y)
            kNN_spectral_pred_Y_abs_self = model.predict(kNN_train_spectral_F_abs)
            kNN_spectral_pred_Y_abs = model.predict(kNN_test_spectral_F_abs)

            model.fit(kNN_train_spectral_F_rel, train_Y)
            kNN_spectral_pred_Y_rel_self = model.predict(kNN_train_spectral_F_rel)
            kNN_spectral_pred_Y_rel = model.predict(kNN_test_spectral_F_rel)

            # leiden relative and absolute
            # model.fit(kNN_train_leiden_F_abs, train_Y)
            # kNN_leiden_pred_Y_abs_self = model.predict(kNN_train_leiden_F_abs)
            # kNN_leiden_pred_Y_abs = model.predict(kNN_test_leiden_F_abs)

            # model.fit(kNN_train_leiden_F_rel, train_Y)
            # kNN_leiden_pred_Y_rel_self = model.predict(kNN_train_leiden_F_rel)
            # kNN_leiden_pred_Y_rel = model.predict(kNN_test_leiden_F_rel)

            test_Ys = [kNN_spectral_pred_Y_abs, kNN_spectral_pred_Y_rel]  # , kNN_leiden_pred_Y_rel, kNN_leiden_pred_Y_abs
            test_Ys_self = [kNN_spectral_pred_Y_abs_self, kNN_spectral_pred_Y_rel_self]  # , kNN_leiden_pred_Y_rel_self, kNN_leiden_pred_Y_abs_self
            N = [n_kNN_train_spectral, n_kNN_train_spectral]  # n_kNN_train_leiden, n_kNN_train_leiden
            kinds_C = ['spectral', 'spectral']  # , 'leiden', 'leiden'
            kinds_F = ['abs', 'rel']  # , 'abs', 'rel'

            for pred_Y, pred_Y_self, n, ck, fk in zip(test_Ys, test_Ys_self, N, kinds_C, kinds_F):
                entry = ['test', k_, ck, fk, n,
                         balanced_accuracy_score(test_Y, pred_Y),
                         f1_score(test_Y, pred_Y),
                         precision_score(test_Y, pred_Y),
                         recall_score(test_Y, pred_Y),
                         recall_score(test_Y, pred_Y, pos_label=0)]
                performance.append(entry)

                entry_self = ['train', k_, ck, fk, n,
                              balanced_accuracy_score(train_Y, pred_Y_self),
                              f1_score(train_Y, pred_Y_self),
                              precision_score(train_Y, pred_Y_self),
                              recall_score(train_Y, pred_Y_self),
                              recall_score(train_Y, pred_Y_self, pos_label=0)]
                self_performance.append(entry_self)

            # adjusted rand index
            full_P = np.unique([x[0] for x in reduced_df.index])
            train_P = full_P[train_I]
            test_P = full_P[test_I]

            train_SI = [get_patient_indices(PID, reduced_df) for PID in train_P]
            train_SI_flat = [ix for sublist in train_SI for ix in sublist]

            test_SI = [get_patient_indices(PID, reduced_df) for PID in test_P]
            test_SI_flat = [ix for sublist in test_SI for ix in sublist]

            joined_SI = np.concatenate((train_SI_flat, test_SI_flat))
            first_Cs = [kNN_train_spectral_C, kNN_train_spectral_C]  # , kNN_train_leiden_C, kNN_train_leiden_C
            second_Cs = [kNN_test_spectral_C_abs, kNN_test_spectral_C_rel]  # , kNN_test_leiden_C_abs, kNN_test_leiden_C_rel

            for C1, C2, ck, fk in zip(first_Cs, second_Cs, kinds_C, kinds_F):
                joined_C = np.concatenate((C1, C2), axis=None)
                ordered_joined_C = np.zeros(len(joined_C))

                for ix, c in zip(joined_SI, joined_C):
                    ordered_joined_C[ix] = c

                if ck == 'spectral':
                    ari.append([ck, fk, k_, adjusted_rand_score(full_spectral_C, ordered_joined_C), adjusted_rand_score(kNN_full_spectral_C, ordered_joined_C)])
                # if ck == 'leiden':
                #     ari.append([ck, fk, k_, adjusted_rand_score(full_leiden_C, ordered_joined_C), adjusted_rand_score(kNN_full_leiden_C, ordered_joined_C)])

            ari.append(['FULL', 'FULL', k_, adjusted_rand_score(full_spectral_C, kNN_full_spectral_C), adjusted_rand_score(full_spectral_C, kNN_full_spectral_C)])  # , adjusted_rand_score(full_leiden_C, kNN_full_leiden_C)

    # results to file
    performance_df = pd.DataFrame(performance, columns=['TYPE', 'K', 'CK', 'FK', 'NC', 'BA', 'F1', 'PR', 'SP', 'SN'])
    self_performance_df = pd.DataFrame(self_performance, columns=['TYPE', 'K', 'CK', 'FK', 'NC', 'BA', 'F1', 'PR', 'SP', 'SN'])
    ari_df = pd.DataFrame(ari, columns=['CK', 'FK', 'K', 'FvJ', 'KvJ'])

    performance_df.to_csv('{}/{}_k_run_test_performance.csv'.format(sm_name, sm_name))
    self_performance_df.to_csv('{}/{}_k_run_train_performance.csv'.format(sm_name, sm_name))
    ari_df.to_csv('{}/{}_k_run_ari.csv'.format(sm_name, sm_name))


if __name__ == '__main__':
    t0 = time.time()
    os.environ['NUMEXPR_MAX_THREADS'] = '52'
    numexpr.set_num_threads(52)

    names = [('PAM70', pam70)]

    for name, path in names:
        print('Let\'s go', name)
        kNN_main(path, name)
        # gamma_main(path, name, 'absolute')

    time_passed = (time.time() - t0) / 3600
    print('Time passed {:.2f}h'.format(time_passed))
    print(name, ' done.')
