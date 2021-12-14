import resource

import networkit
import numpy as np
import pandas as pd
import os
import errno
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, log_loss, matthews_corrcoef
# from sklearn.feature_selection import RFECV, RF# E

# from pipeline import Data

# # # # # # # # #
# M E T H O D S #
# # # # # # # # #


def morisita_horn(x, y, s):
    z = 0
    n = 0
    m = 0
    x_len = len(x)
    y_len = len(y)
    for unique_sequence in s:
        z = z + x.count(unique_sequence) * y.count(unique_sequence)
        n = n + x.count(unique_sequence) ** 2
        m = m + y.count(unique_sequence) ** 2
    C = 2 * z / (((n / x_len ** 2) + (m / y_len ** 2)) * x_len * y_len)
    return C


# # # # # # # # # # # # #
# D I R E C T O R I E S #
# # # # # # # # # # # # #

data_root = '/home/ubuntu/Enno/gammaDelta/patient_data/'

data_bl = data_root + 'BL/'
data_fu = data_root + 'FU/'
data_hd = data_root + 'HD/'
data_si = data_root + 'sick/'

P1003_BL = 'CopyOfVDJTOOLS_.1003_2553_SA78_S78_L001_R1.txt'  # BL_PATIENT_1.fasta
P1004_BL = 'CopyOfVDJTOOLS_.1004_2554_SA79_S79_L001_R1.txt'  # BL_PATIENT_2.fasta
P1003_BL_4_1 = 'VDJTOOLS_.1003_BL_4-1-TCRD_S36_L001_R1.txt'  # BL_PATIENT_49.fasta
P1004_BL_4_3 = 'VDJTOOLS_.1004_BL_4-3-TCRD_S38_L001_R1.txt'  # BL_PATIENT_50.fasta

P1003_FU = 'CopyOfVDJTOOLS_.1003_2564_SA92_S92_L001_R1.txt'  # FU_PATIENT_1.fasta
P1004_FU = 'CopyOfVDJTOOLS_.1004_2568_SA93_S93_L001_R1.txt'  # FU_PATIENT_2.fasta
P1003_FU_4_2 = 'VDJTOOLS_.1003_FU_4-2-TCRD_S37_L001_R1.txt'  # FU_PATIENT_44.fasta
P1004_FU_4_4 = 'VDJTOOLS_.1004_FU_4-4-TCRD_S39_L001_R1.txt'  # FU_PATIENT_45.fasta

dm_root = '/home/ubuntu/Enno/mnt/volume/dm_in_use/'

b45 = 'BLFUHD_BLOSUM45_1_0.1_DM'
b62 = 'BLFUHD_BLOSUM62_10_0.5_DM'
pam70 = 'BLFUHD_PAM70_1_0.1_DM'

b45_gammas = [1.00, 1.05, 1.10, 1.15, 1.16, 1.17]

# # # # # # # # # # # # # #
# D O U B L E S O R T E D #
# # # # # # # # # # # # # #

# eCRF : BL  : FU   :
# 1003 : 4-1 : 4-2  :
# 1004 : 4-3 : 4-4  :
double_eCRF = [1003, 1004]

ds_pairs = [(P1003_BL, P1004_FU), (P1004_BL, P1004_FU), (P1003_BL_4_1, P1003_FU_4_2), (P1004_BL_4_3, P1004_FU_4_4)]
ds = [(P1003_BL, P1003_BL_4_1), (P1004_BL, P1004_BL_4_3), (P1003_FU, P1003_FU_4_2), (P1004_FU, P1004_FU_4_4)]


def double_sorted_overlaps():
    temp_df = pd.DataFrame(columns=['len_s1', 'len_s2', 'len_intersection', 'MHI', 'OV'])

    for sample_1, sample_2 in ds:
        sample_1 = pd.read_csv(data_si + sample_1, sep='\t')
        sample_2 = pd.read_csv(data_si + sample_2, sep='\t')

        s1_cdr = sample_1['cdr3aa']
        s2_cdr = sample_2['cdr3aa']

        intersection = list(set(sample_1['cdr3aa']).intersection(set(sample_2['cdr3aa'])))
        mhi = morisita_horn(s1_cdr.to_list(), s2_cdr.to_list(), list(np.unique(s1_cdr.to_list() + s2_cdr.to_list())))

        l1, l2, li, mhi, ov = len(s1_cdr), len(s2_cdr), len(intersection), '%.3f' % mhi, '%.3f' % (len(intersection) / min(len(s1_cdr), len(s2_cdr)))
        findings = np.array([l1, l2, li, mhi, ov])
        temp_df.loc[len(temp_df)] = findings

    return temp_df


# # # # # # # # # # # # # # #
# C O M P A R E M O D E L S #
# # # # # # # # # # # # # # #

def parse_all_sequences():
    with open('BLFUHD_ALL_SEQUENCES.fasta', 'r') as file:
        lines = file.readlines()

    headers = []
    seq = []
    for line in lines:
        if line.startswith('>'):
            headers.append(line)
        else:
            seq.append(line.strip())

    return np.array(headers), np.array(seq)


def check_ecrf(path_to_txt, kind='BL', print_out=True):
    headers, seq = parse_all_sequences()

    seq = list(zip(headers, seq))
    seq = [(h.strip(), s.strip()) for h, s in seq]

    lo_files = os.listdir(path_to_txt)
    lo_files.sort()

    lo_patient_df = []
    for ix, p in enumerate(lo_files):
        if p.startswith('C'):
            ecrf = p[16:20]
        else:
            ecrf = p[10:14]
        patient = '{}_PATIENT_{}'.format(kind, (ix + 1))
        lo_patient_df.append((pd.read_csv(path_to_txt + p, sep='\t'), ecrf, patient))

    for df, eCRF, patient in lo_patient_df:
        ixs = get_patient_indices(patient)
        if list(np.array(seq)[ixs][:, 1]) == df['cdr3aa'].to_list():
            if print_out:
                print(patient + ' - ' + eCRF)
        else:
            raise AssertionError('Incompatible eCRF')

    return lo_patient_df


def reduce_dm(patient, dm_filename):
    """
    Remove patient from the dm and return the updated dm.
    :param patient:
    :param dm_filename:
    :return:
    """

    dm_path = dm_root + dm_filename
    if os.path.isfile(dm_path):
        dm = joblib.load(dm_path)
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), dm_root+b45)

    for p in patient:
        patient = p + '_'
        indices = get_patient_indices(patient)

        # np.delete(matrix, ix, axis: 0 = rows, 1 = columns)
        dm = np.delete(dm, indices, axis=0)
        dm = np.delete(dm, indices, axis=1)

    return dm


def get_patient_indices(patient: str):
    """
    Determines all sequence indices for a single patient.
    :param patient: patient
    :return:
    """
    if not patient.endswith('_'):
        patient = patient + '_'

    headers, seq = parse_all_sequences()

    ixs = [ix for ix, header in enumerate(headers) if patient in header]

    return ixs

# # # # # # # # # # # # # # # #
# C L A S S I F I C A T I O N #
# # # # # # # # # # # # # # # #


class Classification:
    def __init__(self, dm: np.array, sm: str, gp: tuple, n: int, n_healthy: int, n_sick: int,
                 bl: list, fu: list, hd: list):
        # basics
        self.n_healthy = n_healthy
        self.n_sick = n_sick
        self.n = n
        self.distance_matrix = dm
        self.substitution_matrix = sm
        self.gap_penalties = gp
        # patient data - (df, eCRF, ID)
        self.bl = bl
        self.fu = fu
        self.hd = hd

        # louvain
        self.graph = []
        self.clusters = []
        # logistic regression
        self.feature_vector = []
        self.response = np.array(self.n_sick * [1] + self.n_healthy * [0]).T
        # downstream
        self.positive_significant_feature = []
        self.negative_significant_feature = []

    def dm_to_graph(self):
        """
        Transforms the distance matrix dm into a graph g on which the Louvain method can be performed.
        """

        counter = 0
        m, _ = self.distance_matrix.shape
        g = networkit.Graph(m, weighted=True)

        mask_x, mask_y = np.mask_indices(m, np.tril, -1)
        masking_zip = zip(mask_x, mask_y, self.distance_matrix[mask_x, mask_y])

        for nodeA, nodeB, weight in masking_zip:
            counter += 1
            if counter % (10 ** 7) == 0:
                print(counter)
                print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            g.addEdge(nodeA, nodeB, weight)

        self.graph = g

    def louvain(self, gammas):
        first_loop = True
        for g in gammas:

            cluster = networkit.community.detectCommunities(self.graph,
                                                            algo=networkit.community.PLM(self.graph, refine=True,
                                                                                         gamma=g))
            cluster_vector = cluster.getVector()
            number_of_clusters = len(np.unique(cluster_vector))
            dic = dict(zip(np.unique(cluster_vector), range(number_of_clusters)))
            cluster_vector = [dic[i] for i in cluster_vector]

            if first_loop:
                self.clusters.append(cluster_vector)
                first_loop = False
            else:
                increment = len(np.unique(cluster_vector))
                adjusted_cluster_vector = [x+increment for x in cluster_vector]
                self.clusters.append(adjusted_cluster_vector)

    def build_feature_from_cluster(self, kind='absolute'):

        if kind not in ['absolute', 'relative', 'freq']:
            raise ValueError('\'kind\' has to be either \'absolute\', \'relative\' or \'freq\'')

        sequences_per_clustering = []
        first_loop = True

        for cluster in self.clusters:  # for every clustering
            cluster = np.array(cluster)
            patient_ix = 0

            num_cluster = len(np.unique(cluster))
            sequence_distribution = np.zeros((self.n, num_cluster))
            sequences_per_cluster = [[] for _ in range(num_cluster)]

            for cohort in [self.bl, self.fu, self.hd]:  # for every cohort

                for df, ecrf, tag in cohort:  # for every patient within cohort
                    indices = get_patient_indices(tag)
                    dic = dict(zip(np.unique(cluster), range(num_cluster)))

                    patient_sequences = np.array(df['cdr3aa'].to_list())
                    patient_frequencies = np.array(df['freq'].to_list())

                    patient_clusters = cluster[indices]
                    adjusted_patient_cluster = [dic[i] for i in patient_clusters]

                    for s, c, f in zip(patient_sequences, adjusted_patient_cluster, patient_frequencies):
                        if kind == 'relative' or kind == 'absolute':
                            sequence_distribution[patient_ix, c] += 1
                        if kind == 'freq':
                            sequence_distribution[patient_ix, c] += f

                        sequences_per_cluster[c].append(s)

                    patient_ix += 1
                sequences_per_clustering.append(sequences_per_cluster)

            if first_loop:
                self.feature_vector = sequence_distribution
                first_loop = False
            else:
                self.feature_vector = np.concatenate((self.feature_vector, sequence_distribution), axis=1)

        if kind == 'relative':
            self.feature_vector = self.feature_vector / self.feature_vector.sum(axis=0)

    def filter_colinearity(self):
        corr = np.corrcoef(self.feature_vector, rowvar=False)
        w, v = np.linalg.eig(corr)

        n, m = self.feature_vector.shape

        for ix, eigen_value in enumerate(w):
            if eigen_value < 0.01:
                self.feature_vector = np.delete(self.feature_vector, ix, axis=1)

        j, k = self.feature_vector.shape
        print(str(m - k), ' features were colinear and thus deleted.')

    def select_features(self, upper):
        models = dict()
        for i in range(8, upper):
            rfe = RFE(estimator=LogisticRegression(max_iter=500000, C=10000000), n_features_to_select=i)
            model = LogisticRegression(max_iter=500000, C=10000000)
            models[str(i)] = Pipeline(steps=[('s', rfe), ('m', model)])

        results, names = [], []
        for name, model in models.items():
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
            scores = cross_val_score(model, self.feature_vector, self.response, scoring='balanced_accuracy', cv=cv,
                                     n_jobs=-1, error_score='raise')
            results.append(scores)
            names.append(name)
            print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))

        plt.boxplot(results, labels=names, showmeans=True)
        plt.ylabel('Cross validation score (balanced accuracy)')
        plt.xlabel('Number of features selected')
        plt.title(
            'Feature selection ' + self.substitution_matrix + ' ' + str(self.gap_penalties))
        plt.show()

        # ix = highest scoring index
        max_score = 0
        max_name = ''
        for score, name in zip(scores, names):
            if score > max_score:
                max_score = score
                max_name = name
        ix = max_name

        rfe = models[ix][0]
        rfe.fit(self.feature_vector, self.response)
        mask = rfe.get_support(indices=True)
        self.feature_vector = rfe.transform(self.feature_vector)
        # self.reduced_sequences = [self.reduced_sequences[ix] for ix in mask]
        # self.tracing = [self.tracing[ix] for ix in mask]

    def make_classification(self, reducer=False):
        self.model = LogisticRegressionCV(max_iter=50000)
        self.model_f = self.model.fit(X=self.feature_vector, y=self.response)

        if reducer:
            self.select_features(21)

        cv_sensitivity_score = np.average(np.array(cross_val_score(self.model, self.feature_vector, self.response, cv=5, scoring='recall')))
        # cv_specificity_score = np.average(np.array(cross_val_score(self.model, self.feature_vector, self.response, cv=5, scoring='specificity')))
        cv_roauc_score = np.average(np.array(cross_val_score(self.model, self.feature_vector, self.response, cv=5, scoring='roc_auc')))
        cv_balanced_acc_score = np.average(np.array(cross_val_score(self.model, self.feature_vector, self.response, cv=5, scoring='balanced_accuracy')))
        cv_f1_score = np.average(np.array(cross_val_score(self.model, self.feature_vector, self.response, cv=5, scoring='f1')))
        cv_precision_score = np.average(np.array(cross_val_score(self.model, self.feature_vector, self.response, cv=5, scoring='precision')))

        print('Sensitivity: \t %.3f' % cv_sensitivity_score)
        # print('Specificity: \t %.3f' % cv_specificity_score)
        print('ROAUC: \t %.3f' % cv_roauc_score)

        print('F1: \t %.3f' % cv_f1_score)
        print('Precision: \t %.3f' % cv_precision_score)
        print('Bal. Accuracy: \t %.3f' % cv_balanced_acc_score)


# TODO Wald test
# TODO feature analysis
# TODO UMAP embedding

# # # # # # # # # #
# T H O U G H T S #
# # # # # # # # # #

# BL_PATIENT_1      -       FU_PATIENT_1        -       1003        -       I
# BL_PATIENT_2      -       FU_PATIENT_2        -       1004        -       II
# BL_PATIENT_49     -       FU_PATIENT_44       -       1003_2      -       III
# BL_PATIENT_50     -       FU_PATIENT_45       -       1004_2      -       IV

# 4 variants
# 1. all
# 2. none
# 3. only  I and II
# 4. only II and IV


if __name__ == '__main__':

    double_sorted_df = double_sorted_overlaps()
    print(double_sorted_df)

    # Init
    obj = Classification(dm=joblib.load(dm_root+b45), sm='BLOSUM45', gp=(1, 0.1), n=150,
                         bl=check_ecrf(data_bl, 'BL', print_out=False),
                         fu=check_ecrf(data_fu, 'FU', print_out=False),
                         hd=check_ecrf(data_hd, 'HD', print_out=False),
                         n_healthy=29,
                         n_sick=121)

    obj.dm_to_graph()
    obj.louvain(b45_gammas)

    print('ABSOLUTE')
    obj.build_feature_from_cluster(kind='absolute')
    print(obj.feature_vector.shape)
    obj.make_classification()

    print('RELATIVE')
    obj.build_feature_from_cluster(kind='relative')
    print(obj.feature_vector.shape)
    obj.make_classification()

    print('FREQ')
    obj.build_feature_from_cluster(kind='freq')
    print(obj.feature_vector.shape)
    obj.make_classification()

    print('ABSOLUTE')
    obj.build_feature_from_cluster(kind='absolute')
    obj.make_classification(reducer=True)

    print('RELATIVE')
    obj.build_feature_from_cluster(kind='relative')
    obj.make_classification(reducer=True)

    print('ABSOLUTE')
    obj.build_feature_from_cluster(kind='freq')
    obj.make_classification(reducer=True)


    var_i = []
    var_ii = ['BL_PATIENT_1', 'BL_PATIENT_2', 'BL_PATIENT_49', 'BL_PATIENT_50', 'FU_PATIENT_1', 'FU_PATIENT_2', 'FU_PATIENT_44', 'FU_PATIENT_45']
    var_iii = ['BL_PATIENT_49', 'FU_PATIENT_44', 'BL_PATIENT_50', 'FU_PATIENT_45']
    var_iv = ['BL_PATIENT_1', 'BL_PATIENT_2', 'FU_PATIENT_1', 'FU_PATIENT_2']

    DM = reduce_dm(var_ii, b45)
    print(DM.shape)

"""
self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.feature_vector, self.response,
                                                                                test_size=split, stratify=self.response,
                                                                                random_state=rnd_state)
"""
