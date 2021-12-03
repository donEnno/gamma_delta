"""

- Ähnlichkeit der Doppelten

Modelle:
- B45, B62, PAM beste Modelle
	+ absolut vs relativ
	+ freq Modelle

	- hierfür 4 Varianten
		+ Wie verteilen sich diedoppelten Patienten im Clustering?
		+ Häufigkeiten der Doppelten über die Cluster

Erst nach den vier Varianten:
- BL vs HD (für die 3)

- Response

- Regression der PFS/OS

Wichtigkeit: OS -> PFS -> Response

@sig. Cluster
- Eigenschaften der Cluster:
	+ unique Seq
	+ Patienten
	+ Delta-Ketten

- BL vs FU (matching)

- gemeinsame Sequenzen


"""
import resource

import networkit
import numpy as np
import pandas as pd
import os
import errno
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, log_loss, matthews_corrcoef
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import RFECV, RFE

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

P1003_BL = 'CopyOfVDJTOOLS_.1003_2553_SA78_S78_L001_R1.txt'  # BL_PATIENT_1.fasta             \/
P1004_BL = 'CopyOfVDJTOOLS_.1004_2554_SA79_S79_L001_R1.txt'  # BL_PATIENT_2.fasta
P1003_BL_4_1 = 'VDJTOOLS_.1003_BL_4-1-TCRD_S36_L001_R1.txt'  # BL_PATIENT_49.fasta
P1004_BL_4_3 = 'VDJTOOLS_.1004_BL_4-3-TCRD_S38_L001_R1.txt'  # BL_PATIENT_50.fasta

P1003_FU = 'CopyOfVDJTOOLS_.1003_2564_SA92_S92_L001_R1.txt'  # FU_PATIENT_1.fasta
P1004_FU = 'CopyOfVDJTOOLS_.1004_2568_SA93_S93_L001_R1.txt'  # FU_PATIENT_2.fasta             \/
P1003_FU_4_2 = 'VDJTOOLS_.1003_FU_4-2-TCRD_S37_L001_R1.txt'  # FU_PATIENT_44.fasta
P1004_FU_4_4 = 'VDJTOOLS_.1004_FU_4-4-TCRD_S39_L001_R1.txt'  # FU_PATIENT_45.fasta            \/

dm_root = '/home/ubuntu/Enno/mnt/volume/dm_in_use/'

b45 = 'BLFUHD_BLOSUM45_1_0.1_DM'
b62 = 'BLFUHD_BLOSUM62_10_0.5_DM'
pam70 = 'BLFUHD_PAM70_1_0.1_DM'

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
            seq.append(line)

    return headers, seq


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
    with open('BLFUHD_ALL_SEQUENCES.fasta', 'r') as file:
        lines = file.readlines()

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
    def __init__(self, dm: np.array, sm: str, gp: tuple,
                 bl: list, fu: list, hd: list):
        # basics
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
        self.response = []
        # downstream
        self.positive_significant_feature = []
        self.negative_significant_feature = []

    def dm_to_graph(self, dm):
        """
        Transforms the distance matrix dm into a graph g on which the Louvain method can be performed.
        :param dm: distance matrix in lower triangular format
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

    def louvain(self, graph, gammas):
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

    def build_feature_from_cluster(self):

        number_of_sequences_per_patient = []
        for cohort in [self.bl, self.fu, self.hd]:
            cohort_temp = []
            for df, ecrf, tag in cohort:
                n, _ = df.shape
                cohort_temp.append(n)

                indices = get_patient_indices(tag)


            number_of_sequences_per_patient.append(cohort_temp)


        # TODO Maybe use freq as increment instead of 1
        # TODO for loop for each clustering
        frequency = np.zeros(self.clusters[0])

        for aa, (s, c) in zip(aa_sequences, patient_list):
            if frequency[c] == 0:
                frequency += 1
            else:
                frequency[c] += 1
            sequences_per_cluster[c].append(aa)

        return frequency, sequences_per_cluster

        def calculate_feature_vector(self, absolute_toggle=True):
            absolute = []
            relative = []
            num_of_seq_per_patient = []

            patient_types = self.split_origin_to_types()

            for typ in patient_types:
                num_of_seq_per_patient.extend(self.get_number_of_sequences_per_patient(typ))

            sequences_per_cluster = [[] for _ in range(self.number_of_clusters)]

            dic = dict(zip(np.unique(self.cluster_vector), range(self.number_of_clusters)))
            new_cluster_vector = [dic[i] for i in self.cluster_vector]
            self.cluster_vector = new_cluster_vector

            cluster_tuple_list = list(zip(range(self.number_of_sequences), self.cluster_vector))

            upper = 0
            for chunk in num_of_seq_per_patient:
                lower = upper
                upper += chunk

                # Split partition and list_of_sequences in [lower:upper] where [l:u] is the range of sequences for one patient.
                temp_freq, sequences_per_cluster = self.count_frequency_for_one_patient(cluster_tuple_list[lower:upper],
                                                                                        self.sequences[lower:upper],
                                                                                        sequences_per_cluster)
                temp_sum = sum(temp_freq)
                absolute.append(temp_freq)
                relative.append(temp_freq / temp_sum)

            if absolute_toggle:
                self.feature_vector = np.array(absolute)
            else:
                self.feature_vector = np.array(relative)
            self.sequences_per_cluster = sequences_per_cluster


# TODO Classification w/ feature selection
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
    obj = Classification(dm=joblib.load(dm_root+b45), sm='BLOSUM45', gp=(1, 0.1),
                         bl=check_ecrf(data_bl, 'BL', print_out=False),
                         fu=check_ecrf(data_fu, 'FU', print_out=False),
                         hd=check_ecrf(data_hd, 'HD', print_out=False)
                         )

    var_i = []
    var_ii = ['BL_PATIENT_1', 'BL_PATIENT_2', 'BL_PATIENT_49', 'BL_PATIENT_50', 'FU_PATIENT_1', 'FU_PATIENT_2', 'FU_PATIENT_44', 'FU_PATIENT_45']
    var_iii = ['BL_PATIENT_49', 'FU_PATIENT_44', 'BL_PATIENT_50', 'FU_PATIENT_45']
    var_iv = ['BL_PATIENT_1', 'BL_PATIENT_2', 'FU_PATIENT_1', 'FU_PATIENT_2']

    DM = reduce_dm(var_ii, b45)
    print(DM.shape)
