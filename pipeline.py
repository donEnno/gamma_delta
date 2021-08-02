# Imports
# Defaults
import math
import os
import errno
import time
from joblib import Parallel, delayed, dump, load
import pandas as pd
import numpy as np
import sys
from itertools import combinations, product

# Alignments
from Bio import SeqIO, pairwise2
from Bio.Align import substitution_matrices
from Bio import Phylo

# Projection
import seaborn as sns
import umap
import umap.plot
import networkit
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ML
from sklearn.linear_model import LogisticRegression     # , LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split    # , StratifiedShuffleSplit
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


class Data:

    def __init__(self,
                 origin='BLHD',
                 substitution_matirx='BLOSUM45',
                 gap_open=10,
                 gap_extend=0.5,
                 ):

        # Root
        self.performance = []
        self.gd_root = fr"/home/ubuntu/Enno/gammaDelta/sequence_data/"
        self.mnt_root = fr'/home/ubuntu/Enno/mnt/volume/'

        # Alignment Parameter
        self.substitution_matrix = substitution_matirx
        self.gap_open = gap_open
        self.gap_extend = gap_extend

        # Sequence Data
        self.sequences = []
        self.number_of_sequences = 0
        self.dm = np.array([])
        self.dataframe = []

        # Directories
        self.origin = origin
        self.fasta_location = ''
        self.dm_location = ''
        self.cluster_location = ''
        self.plot_location = ''

        # Cluster
        self.gamma = 0
        self.embedding = []
        self.graph = networkit.Graph()
        self.cluster = []
        self.cluster_vector = []
        self.number_of_clusters = 0

        # Features
        self.feature_vector = []
        self.sequences_per_cluster = []

        # Classification
        self.response = []
        self.model = LogisticRegression()
        self.test_x = []
        self.test_y = []
        self.train_x = []
        self.train_y = []
        self.regularization_c = 0
        self.model_score = 0
        self.model_report = []
        self.auc = 0
        self.coef = []
        self.highest_coef = 0
        self.lowest_coef = 0
        self.z = []

        # Analysis
        self.cluster_of_interest_one = []
        self.cluster_of_interest_zero = []

    # # # # # # # # # # #
    #   SETTER METHODS  #
    # # # # # # # # # # #

    def set_fasta_location(self):

        fasta_loc = self.gd_root + self.origin + '_fasta/' + self.origin + '_ALL_SEQUENCES.fasta'
        self.fasta_location = fasta_loc

    def set_dm_location(self):
        self.dm_location = self.mnt_root + 'distance_matrices/' + self.origin + '_' + self.substitution_matrix + '_' + str(self.gap_open) + '_' + str(self.gap_extend) + '_DM'

    def set_cluster_location(self):

        self.cluster_location = self.mnt_root + 'cluster/' + self.origin + '_' + self.substitution_matrix + '_' + str(self.gap_open) + '_' + str(self.gap_extend) + '_' + str(self.gamma) + '_C'

    def set_plot_location(self):
        self.plot_location = self.gd_root + 'plots/' + self.origin + '_' + str(self.gamma)

    def set_sequences(self):
        self.sequences = list(enumerate(SeqIO.parse(self.fasta_location, "fasta")))

    def set_num_seq(self):

        self.number_of_sequences = len(self.sequences)

    def set_data_frame(self):

        ix, iy = self.dm.shape
        self.dataframe = pd.DataFrame(data=self.dm, index=[f'Sequence_{i}' for i in range(1, ix + 1)],
                                      columns=[f'Sequence_{i}' for i in range(1, iy + 1)])

    # # # # # # # # # # # #
    #   DISTANCE MATRIX   #
    # # # # # # # # # # # #

    def generate_batches_from_file(self):

        n_of_pairs = 0
        for i in self.sequences:
            n_of_pairs += i[0]

        n_jobs = 27
        sequences_per_job = math.ceil(n_of_pairs / n_jobs)

        job_batches = []
        temp_batch = []
        temp_sum = 0

        for ix in self.sequences:
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

    def calculate_distance_matrix(self, identity=False):

        output_filename_memmap = '/home/ubuntu/Enno/gammaDelta/joblib_memmap/output_memmap'
        output = np.memmap(output_filename_memmap, dtype=float, mode='w+',
                           shape=(self.number_of_sequences, self.number_of_sequences))

        batches = self.generate_batches_from_file()

        Parallel(n_jobs=-1, verbose=50)(delayed(self.compute_pairwise_scores)(identity, batch, output) for batch in batches)
        dump(output, self.dm_location)

    def convert_newick_to_distance_matrix(self):
        # sys.setrecursionlimit(10000)          -- >        needed?

        file = '/home/ubuntu/Enno/gammaDelta/sequence_data/BLHD_fasta/1_BLHD_TREE2.phy'
        tree = Phylo.read(file, 'newick')

        seq_names = tree.get_terminals()
        dmat = {}

        # length = sum(1 for _ in combinations(seq_names, 2))
        # print(length)       -->     437887621

        counter = 0
        timer = time.time()

        for p1, p2 in combinations(seq_names, 2):
            counter += 1
            if counter % 10 ** 2 == 0:
                print(counter)
                lap = time.time()
                timing = lap - timer
                print('This took %.f2 [s]' % timing)
                break

            d = tree.distance(p1, p2)
            dmat[(p1.name, p2.name)] = d

    def set_dm(self):

        if os.path.isfile(self.dm_location):
            self.dm = load(self.dm_location)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.dm_location)

    def compute_pairwise_scores(self, identity: bool, batch: list, output):

        matrix = substitution_matrices.load(self.substitution_matrix)

        for seq_a in batch:
            for seq_b in self.sequences:
                if seq_a[0] > seq_b[0]:

                    if not identity:
                        res_ = pairwise2.align.globalds(seq_a[1].seq, seq_b[1].seq, matrix, -self.gap_open,
                                                        -self.gap_extend, score_only=True)
                    else:
                        res_ = pairwise2.align.globalxx(seq_a[1].seq, seq_b[1].seq, score_only=True)

                    output[seq_a[0]][seq_b[0]] = res_
                else:
                    continue

    # # # # # # # # # # # # #
    #  UMAP AND CLUSTERING  #
    # # # # # # # # # # # # #

    def set_embedding(self, spread, min_dist, a, b):

        sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})

        reducer = umap.UMAP()
        self.embedding = reducer.fit_transform(self.dataframe)  # type(embedding) = <class 'numpy.ndarray'

    def set_graph(self):

        timer = time.time()
        counter = 0
        m, _ = self.dm.shape
        g = networkit.Graph(m, weighted=True)

        mask_x, mask_y = np.mask_indices(m, np.tril, -1)
        masking_zip = zip(mask_x, mask_y, self.dm[mask_x, mask_y])

        for nodeA, nodeB, weight in masking_zip:
            counter += 1
            if counter % (10 ** 7) == 0:
                print(counter)
            g.addEdge(nodeA, nodeB, weight)

        self.graph = g
        print('This took %.3f' % (time.time()-timer))

    def calculate_communities(self, save_cluster=False):

        cluster = networkit.community.detectCommunities(self.graph,
                                                        algo=networkit.community.PLM(self.graph, refine=True, gamma=self.gamma))
        cluster_vector = cluster.getVector()
        self.cluster = cluster
        self.cluster_vector = cluster_vector
        self.number_of_clusters = len(np.unique(self.cluster_vector))

        if save_cluster:
            dump(cluster_vector, self.cluster_location)

    def set_cluster(self):

        self.cluster = load(self.cluster_location)
        self.cluster_vector = self.cluster.getVector()
        self.number_of_clusters = len(np.unique(self.cluster_vector))

        dic = dict(zip(np.unique(self.cluster_vector), range(self.number_of_clusters)))
        new_cluster_vector = [dic[i] for i in self.cluster_vector]
        self.cluster_vector = new_cluster_vector

    def plot_cluster(self, save_fig=False):

        cmap = cm.get_cmap('prism', max(self.cluster_vector) + 1)
        x = self.embedding[:, 0]
        y = self.embedding[:, 1]
        plt.scatter(x, y, cmap=cmap, c=list(self.cluster_vector), s=5, alpha=0.5)
        plt.title(f'Louvain com. det. in UMAP projection, patient type: ' + self.origin + ' using gamma ' + str(self.gamma), fontsize=15)
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.show()
        if save_fig:
            plt.savefig(self.plot_location)

    # # # # # # # # # # # #
    #   FEATURE BUILDING  #
    # # # # # # # # # # # #

    def split_origin_to_types(self):

        types = [self.origin[i:i + 2] for i in range(0, len(self.origin), 2)]
        return types

    def get_number_of_sequences_per_patient(self, patient_type):

        list_of_num_seq = []
        path_to_seqs_per_origin = fr'/home/ubuntu/Enno/gammaDelta/sequence_data/{patient_type}_fasta/'

        num_of_patients = len(os.listdir(path_to_seqs_per_origin))
        num_of_patients = range(1, num_of_patients+1)

        for patient_number in num_of_patients:
            file = fr'{path_to_seqs_per_origin}{patient_type}_PATIENT_{patient_number}.fasta'
            list_of_num_seq.append(len([1 for line in open(file) if line.startswith(">")]))

        return list_of_num_seq

    def count_frequency_for_one_patient(self, patient_list, aa_sequences, sequences_per_cluster):

        frequency = np.zeros(self.number_of_clusters)

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
            temp_freq, sequences_per_cluster = self.count_frequency_for_one_patient(cluster_tuple_list[lower:upper], self.sequences[lower:upper],
                                                                                    sequences_per_cluster)
            temp_sum = sum(temp_freq)
            absolute.append(temp_freq)
            relative.append(temp_freq / temp_sum)

        if absolute_toggle:
            # return np.array(absolute), sequences_per_cluster
            self.feature_vector = np.array(absolute)
        else:
            # return np.array(relative), sequences_per_cluster
            self.feature_vector = np.array(relative)
        self.sequences_per_cluster = sequences_per_cluster

    def set_response(self):

        response = []

        bl = [1 for _ in range(66)]
        fu = [1 for _ in range(55)]
        hd = [0 for _ in range(29)]

        types = [self.origin[i:i + 2] for i in range(0, len(self.origin), 2)]

        for typ in types:
            if typ == 'BL':
                response.extend(bl)
            if typ == 'FU':
                response.extend(fu)
            if typ == 'HD':
                response.extend(hd)

        self.response = response

    # # # # # # # # # # #
    #   MODEL METHODS   #
    # # # # # # # # # # #

    def sk_build_model(self):

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.feature_vector, self.response,
                                                                                test_size=0.2, stratify=self.response, random_state=2)
        self.model = LogisticRegression(solver='lbfgs', n_jobs=-1, C=self.regularization_c, random_state=2, max_iter=5000)
        self.model.fit(self.train_x, self.train_y)

    def sm_build_model(self, solver='newton', rnd_state=2, concat=False):

        if concat:
            self.final_feature = np.concatenate((self.final_feature, self.feature_vector), axis=1)
        else:
            self.final_feature = self.feature_vector

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.final_feature, self.response,
                                                                                test_size=0.2, stratify=self.response,
                                                                                random_state=rnd_state)

        self.model = sm.Logit(self.train_y, self.train_x)
        self.result = self.model.fit(method=solver, maxiter=5000)

        print(self.result.summary())

    def sm_build_reg_model(self, l1_w=0):
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.feature_vector, self.response,
                                                                                test_size=0.2, stratify=self.response,
                                                                                random_state=2)
        self.model = sm.Logit(self.train_y, self.train_x).fit_regularized(L1_wt=l1_w, maxiter=150)
        print(self.model.summary())

    # # # # # # # # # # # # # # # #
    #   MODEL EVALUATION METHODS  #
    # # # # # # # # # # # # # # # #

    def sk_model_evaluation(self):

        self.model_score = self.model.score(self.test_x, self.test_y)

    def sk_set_model_report(self):

        self.model_report = classification_report(self.test_y, self.model.predict(self.test_x))

    def sk_cv_scores(self):

        self.model = LogisticRegression(solver='lbfgs', random_state=2, max_iter=5000)

        self.cv_f1_score = np.average(np.array(cross_val_score(self.model, self.final_feature, self.response, cv=5, scoring='f1')))
        self.cv_rocauc_score = np.average(np.array(cross_val_score(self.model, self.final_feature, self.response, cv=5, scoring='roc_auc')))
        self.cv_prec_score = np.average(np.array(cross_val_score(self.model, self.final_feature, self.response, cv=5, scoring='precision')))
        self.cv_balanced_acc_score = np.average(np.array(cross_val_score(self.model, self.final_feature, self.response, cv=5, scoring='balanced_accuracy')))

        print('CV F1 Score: \t %.3f' % self.cv_f1_score)
        print('CV ROCAUC Score: \t %.3f' %  self.cv_rocauc_score)
        print('CV AVG PREC Score: \t %.3f' %  self.cv_prec_score)
        print('CV AVG BALANCED ACC Score: \t %.3f' %  self.cv_balanced_acc_score)

    def sm_eval(self):
        yhat = self.result.predict(self.test_x)
        prediction = list(map(round, yhat))

        print('Actual values', list(self.test_y))
        print('Predictions :', prediction)
        cm = confusion_matrix(self.test_y, prediction)
        print("Confusion Matrix : \n ", cm[0], '\n', cm[1])
        print('Test accuracy: ', accuracy_score(self.test_y, prediction))
        print('Balanced accuravy: ', balanced_accuracy_score(self.test_y, prediction))
        print('ROC AUC score: ', roc_auc_score(self.test_y, yhat))

    def print_confusion_matrix(self):

        co_ma = confusion_matrix(self.test_y, self.model.predict(self.test_x))
        print(r'A\P', '\t', '0', '\t', '1')
        print(12 * '=')

        for i in range(2):
            print(i, ' \t', co_ma[i, 0], '\t', co_ma[i, 1])
            print(12 * '=')

    def sk_plot_roc(self):

        probabilities = self.model.predict_proba(X=self.test_x)
        probabilities = probabilities[:, 1]

        self.auc = roc_auc_score(self.test_y, probabilities)

        # Calculate roc curves
        lr_fpr, lr_tpr, _ = roc_curve(self.test_y, probabilities)

        # Plot the curve for the model
        plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
        plt.title(label=f'Logistic: ROC AUC=%.3f - C={self.regularization_c}' % self.auc)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()
        plt.clf()

    def sm_feature_plot(self):

    def highlight_clusters(self):

        cmap = mcolors.ListedColormap(["grey", "green", "red"])
        plt.scatter(self.embedding[:, 0], self.embedding[:, 1], c=self.z, cmap=cmap, s=5, alpha=0.25)
        plt.title('Clusters of interest', fontsize=15)
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.show()

    # Todo - Clusteroverlap
    # Todo - Concatenate Featurevectors
    # Todo - Downstream Analysis

        for i, (b1, p1), s1 in self.cluster_of_interest_zero:
            for j, (b2, p2), s2 in self.cluster_of_interest_zero:
                print(type(s1), len(s1))
                print(type(s2), len(s2))
                if s1 == s2:
                    print('ID worked.')
                else:
                    try:
                        seq1 = [seq.seq for seq in s1]
                        seq2 = [seq.seq for seq in s2]
                        overlap = [seq for seq in seq1 if seq in seq2]
                        print('%s sequences overlap, that equals a fraction of %s' % (
                        str(len(overlap)), str(len(overlap) / max(len(s1), len(s2)))))
                        list_of_overlaps_one.append(overlap)
                    except NotImplementedError:
                        print('NotImplementedError avoided.')
                        pass

    def sk_prepare_traceback(self):

            self.coef = self.model.coef_[0]

            if len(self.coef) == self.number_of_clusters:
                print('!!!!!!!!!!!!!!!!!!!!!!!!')

            traceback = zip(self.coef, self.sequences_per_cluster, range(len(self.coef)))
            traceback = list(traceback)
            traceback = sorted(traceback, key=lambda cf: cf[0])

            if self.number_of_clusters > 100:
                self.highest_coef = traceback[-11:-1][-1]
                self.lowest_coef = traceback[0:10][-1]

            else:
                self.highest_coef = traceback[-1][-1]
                self.lowest_coef = traceback[0][-1]


    def sk_set_z(self):

        self.z = []
        for cluster in self.cluster_vector:
            if self.number_of_clusters > 100:
                if cluster in self.highest_coef:
                    self.z.append(1)
                elif cluster in self.lowest_coef:
                    self.z.append(2)
                else:
                    self.z.append(0)
            else:
                if cluster == self.highest_coef:
                    self.z.append(1)
                elif cluster ==  self.lowest_coef:
                    self.z.append(2)
                else:
                    self.z.append(0)

        self.z = np.array(self.z)

    def sk_highlight_clusters(self):

        cmap = mcolors.ListedColormap(["grey", "green", "red"])
        plt.scatter(self.embedding[:, 0], self.embedding[:, 1], c=self.z, cmap=cmap, s=5, alpha=0.25)
        plt.title('Clusters of interest', fontsize=15)
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.show()

    def plot_sequence_distribution(self):

        data = self.feature_vector[-1]
        num_of_cluster = len(data[0])
        temp = [[] for _ in range(num_of_cluster)]

        for p in data:
            for i in range(num_of_cluster):
                temp[i].append(p[i])

    BLHD.set_sequences()
    print(len(BLHD.sequences))
    BLHD.set_num_seq()
    print(BLHD.number_of_sequences)

    sms = ['BLOSUM45', 'BLOSUM62', 'BLOSUM80', 'GONNET1992']

    for subma in sms:

        BLHD.substitution_matrix = subma
        BLHD.dm_location = fr'/home/ubuntu/Enno/mnt/volume/distance_matrices/' + BLHD.origin + '_' + BLHD.substitution_matrix + '_' + str(BLHD.gap_open) + '_' + str(BLHD.gap_extend) + '_DM'

        BLHD.calculate_distance_matrix()
