# Imports
# Defaults
import os
import errno
import resource
import time

import networkit
from joblib import load
import pandas as pd
import numpy as np

# Alignments
from Bio import SeqIO

# Projection
import seaborn as sns
import umap
import umap.plot
# import networkit
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# ML
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, log_loss, matthews_corrcoef
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import RFECV, RFE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# from stability_selection import StabilitySelection


class Data:
    def __init__(self,
                 origin='BLFUHD',
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
        self.number_of_patients = 0
        self.sequences = []
        self.number_of_sequences = 0
        self.dm = np.array([])
        self.dataframe = []
        self.cv_idxs = []
        # Directories
        self.origin = origin
        self.fasta_location = ''
        self.dm_location = ''

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
        self.final_feature = np.array([[] for _ in range(95)])
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

    def set_directories(self, dm,
                        fasta_location='/home/ubuntu/Enno/gammaDelta/sequence_data/BLFUHD_fasta/BLFUHD_ALL_SEQUENCES.fasta'):
        self.dm_location = dm
        self.fasta_location = fasta_location

    def set_dm_properties(self, substitution_matrix, origin='BLFUHD', gap_open=10, gap_extend=0.5):
        self.origin = origin
        self.substitution_matrix = substitution_matrix
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        if os.path.isfile(self.dm_location):
            self.dm = load(self.dm_location)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.dm_location)

    def set_sequence_info(self):
        self.sequences = list(enumerate(SeqIO.parse(self.fasta_location, "fasta")))
        self.number_of_sequences = len(self.sequences)

    def set_basics(self, dm_path, substitution_matrix):
        self.set_directories(dm=dm_path)
        self.set_dm_properties(substitution_matrix=substitution_matrix)
        self.set_sequence_info()

    # # # # # # # # # # # # #
    #  UMAP AND CLUSTERING  #
    # # # # # # # # # # # # #

    def set_data_frame(self):
        ix, iy = self.dm.shape
        self.dataframe = pd.DataFrame(data=self.dm, index=[f'Sequence_{i}' for i in range(1, ix + 1)],
                                      columns=[f'Sequence_{i}' for i in range(1, iy + 1)])

    def set_embedding(self, min_dist): # , spread, min_dist, a, b
        sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})
        reducer = umap.UMAP(min_dist=min_dist) # spread=spread, min_dist=min_dist, a=a, b=b
        self.embedding = reducer.fit_transform(self.dm)  # type(embedding) = <class 'numpy.ndarray'

    def set_graph_v2(self):
        timer = time.time()
        m, _ = self.dm.shape
        g = networkit.Graph(m, weighted=True)
        print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        ix = 0
        counter = 0
        for row in self.dm:
            iy = 0
            for field in row:

                if field > 0:
                    counter += 1
                    g.addEdge(ix, iy, field)
                    if counter % (10 ** 7) == 0:
                        print(counter)
                        print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                iy += 1
            ix += 1
        self.graph = g
        del g
        print('This took %.3f' % (time.time()-timer))

    def set_graph(self):

        timer = time.time()
        counter = 0
        m, _ = self.dm.shape
        print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        g = networkit.Graph(m, weighted=True)
        print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        mask_x, mask_y = np.mask_indices(m, np.tril, -1)
        print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        masking_zip = zip(mask_x, mask_y, self.dm[mask_x, mask_y])
        print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        for nodeA, nodeB, weight in masking_zip:
            counter += 1
            if counter % (10 ** 7) == 0:
                print(counter)
                print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                # gc.collect()
                # objgraph.show_most_common_type()
            g.addEdge(nodeA, nodeB, weight)
            del nodeA, nodeB, weight

        self.graph = g
        print('This took %.3f' % (time.time()-timer))

    def set_graph_heavy_memory(self):
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

    def tril_to_full_dm(self):
        v = self.dm[np.tril_indices(self.dm.shape[0], k=-1)]
        X = np.zeros(self.dm.shape)
        X[np.tril_indices(X.shape[0], k=-1)] = v
        X = X + X.T
        self.dm = X

    def find_cluster(self, gamma):

        cluster = networkit.community.detectCommunities(self.graph,
                                                        algo=networkit.community.PLM(self.graph, refine=True, gamma=gamma))
        cluster_vector = cluster.getVector()
        self.cluster = cluster
        self.cluster_vector = cluster_vector
        self.number_of_clusters = len(np.unique(self.cluster_vector))

        dic = dict(zip(np.unique(self.cluster_vector), range(self.number_of_clusters)))
        new_cluster_vector = [dic[i] for i in self.cluster_vector]
        self.cluster_vector = new_cluster_vector

    def plot_cluster(self, cluster_vector, title, color='viridis'):

        cmap = cm.get_cmap(color, max(cluster_vector) + 1)
        x = self.embedding[:, 0]
        y = self.embedding[:, 1]
        plt.scatter(x, y, cmap=cmap, c=list(cluster_vector), s=5, alpha=0.5)
        plt.title(title, fontsize=15)
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.show()

    # # # # # # # # # # # #
    #   FEATURE BUILDING  #
    # # # # # # # # # # # #

    def split_origin_to_types(self):
        types = [self.origin[i:i + 2] for i in range(0, len(self.origin), 2)]
        return types

    def get_number_of_sequences_per_patient(self, patient_type):
        list_of_num_seq = []
        path_to_seqs_per_origin = fr'/home/ubuntu/Enno/gammaDelta/sequence_data/{patient_type}_fasta/'

        self.number_of_patients = len(os.listdir(path_to_seqs_per_origin))
        num_of_patients = range(1, self.number_of_patients+1)

        for patient_number in num_of_patients:
            file = fr'{path_to_seqs_per_origin}{patient_type}_PATIENT_{patient_number}.fasta'
            list_of_num_seq.append(len([1 for line in open(file) if line.startswith(">")]))

        return list_of_num_seq

    def count_frequency_for_one_patient(self, patient_list, aa_sequences, sequences_per_cluster):
        # TODO Maybe use freq as increment instead of 1

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
            self.feature_vector = np.array(absolute)
        else:
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

    def split_feature(self, rnd_state=2, split=0.2):
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.feature_vector, self.response,
                                                                                test_size=split, stratify=self.response,
                                                                                random_state=rnd_state)

    def split_feature_independent(self, rnd_state=2, split=0.2):
        self.feature_vector, self.independent_x, self.response, self.independent_y = train_test_split(self.feature_vector, self.response,
                                                                                test_size=split, stratify=self.response,
                                                                                random_state=rnd_state)

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.feature_vector, self.response,
                                                                                test_size=split, stratify=self.response,
                                                                                random_state=rnd_state)

    def feature_selection_cv(self, scoring, plot=False):
        min_features_to_select = 4
        rfecv = RFECV(estimator=self.model, step=1, cv=self.cv, scoring=scoring,
                      min_features_to_select=min_features_to_select)
        rfecv.fit(self.feature_vector, self.response)
        self.reduced_feature_vector = rfecv.transform(self.feature_vector)

        rdcd_fv = []
        self.reduced_sequences = []

        idxs = rfecv.get_support(indices=True)
        print(idxs)

        self.tracing = []  # store every traceback vector here

        for ix in idxs:  # loop through slected features indexes

            rdcd_fv.append(self.feature_vector[:,ix])
            self.reduced_sequences.append(self.sequences_per_cluster[ix])

            for run_ix, run in enumerate(self.cluster_vector):
                # loop through every clustering
                if ix in run:  # if selected cluster in this run
                    temp = []  # store selected feature traceback here

                    for s_ix, s in enumerate(run):
                        # correct cv to a format which enables tracback
                        if s == ix:
                            temp.append(0)
                        else:
                            temp.append(1)
                    self.tracing.append(temp)
                    break
                else:
                    continue

        rdcd_fv = np.array(rdcd_fv).T
        print(rdcd_fv==self.reduced_feature_vector)

        if plot:
            plt.figure()
            plt.xlabel("Number of features selected")
            plt.ylabel("Cross validation score (balanced accuracy)")
            plt.title("RFE-CV for " + self.substitution_matrix + ' ' + str(self.gap_open) + '/' + str(self.gap_extend))
            plt.plot(range(min_features_to_select,
                           len(rfecv.grid_scores_) + min_features_to_select),
                     rfecv.grid_scores_)
            plt.show()

    def sk_build_model(self, l1_ratio=None, penalty='l2', solver='lbfgs', reg_c=1000000000000):
        """
        :param penalty {‘l1’, ‘l2’, ‘elasticnet’, ‘none’}, default=’l2’
                Used to specify the norm used in the penalization.
                The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties.
                ‘elasticnet’ is only supported by the ‘saga’ solver.
                If ‘none’ (not supported by the liblinear solver), no regularization is applied.
        :param solver {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default=’lbfgs’
                Algorithm to use in the optimization problem.
                For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.
                For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes.
                ‘newton-cg’, ‘lbfgs’, ‘sag’ and ‘saga’ handle L2 or no penalty
                ‘liblinear’ and ‘saga’ also handle L1 penalty
                ‘saga’ also supports ‘elasticnet’ penalty
                ‘liblinear’ does not support setting penalty='none'
                Note that ‘sag’ and ‘saga’ fast convergence is only guaranteed on features with approximately the same scale. You can preprocess the data with a scaler from sklearn.preprocessing.
        :param l1_ratio The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1.
                Only used if penalty='elasticnet'. Setting l1_ratio=0 is equivalent to using penalty='l2',
                while setting l1_ratio=1 is equivalent to using penalty='l1'.
                For 0 < l1_ratio <1, the penalty is a combination of L1 and L2.
        :param reg_c:
        :return:
        """

        self.model = LogisticRegression(penalty=penalty, solver=solver, l1_ratio=l1_ratio, n_jobs=-1, C=reg_c, random_state=2, max_iter=50000)
        self.result = self.model.fit(self.train_x, self.train_y)

    # # # # # # # # # # # # # # # #
    #   MODEL EVALUATION METHODS  #
    # # # # # # # # # # # # # # # #

    def model_evaluation(self):
        cv_f1_score = np.average(np.array(cross_val_score(self.model, self.feature_vector, self.response, cv=5, scoring='f1')))
        cv_rocauc_score = np.average(np.array(cross_val_score(self.model, self.feature_vector, self.response, cv=5, scoring='roc_auc')))
        cv_prec_score = np.average(np.array(cross_val_score(self.model, self.feature_vector, self.response, cv=5, scoring='precision')))
        cv_balanced_acc_score = np.average(np.array(cross_val_score(self.model, self.feature_vector, self.response, cv=5, scoring='balanced_accuracy')))
        cv_mcc = []
        cv_log_loss = []

        self.cv = StratifiedKFold(n_splits=5)

        for train_index, test_index in self.cv.split(self.feature_vector, self.response):
            self.response = np.array(self.response)
            self.train_x, self.test_x = self.feature_vector[train_index], self.feature_vector[test_index]
            self.train_y, self.test_y = self.response[train_index], self.response[test_index]
            self.sk_build_model()

            y_pred = self.model.predict(self.test_x)
            mcc = matthews_corrcoef(self.test_y, y_pred)
            cv_mcc.append(mcc)

            y_pred_proba = self.model.predict_proba(self.test_x)
            ll = log_loss(self.test_y, y_pred_proba)
            cv_log_loss.append(ll)

        mcc = np.average(np.array(cv_mcc))
        ll = np.average(np.array(cv_log_loss))

        print('MCC Score: \t %.3f' % mcc)
        print('Log-Loss: \t %.3f' % ll)
        print('CV F1 Score: \t %.3f' % cv_f1_score)
        print('CV ROCAUC Score: \t %.3f' %  cv_rocauc_score)
        print('CV AVG PREC Score: \t %.3f' %  cv_prec_score)
        print('CV AVG BALANCED ACC Score: \t %.3f' %  cv_balanced_acc_score)

    def print_confusion_matrix(self):

        co_ma = confusion_matrix(self.test_y, self.model.predict(self.test_x))
        print(r'A\P', '\n', '\t', '0', '\t', '1')
        print(12 * '=')

        for i in range(2):
            print(i, ' \t', co_ma[i, 0], '\t', co_ma[i, 1])
            print(12 * '=')

    def plot_roc(self):

        probabilities = self.model.predict_proba(X=self.test_x)
        probabilities = probabilities[:, 1]

        self.auc = roc_auc_score(self.test_y, probabilities)

        # Calculate roc curves
        lr_fpr, lr_tpr, _ = roc_curve(self.test_y, probabilities)

        # Plot the curve for the model
        plt.plot(lr_fpr, lr_tpr, marker='.')
        plt.title(label=f'ROC AUC=%.3f' % self.auc)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()
        plt.clf()

    def logit_pvalue(self, model, x):
        """ Calculate z-scores for scikit-learn LogisticRegression.
        parameters:
            model: fitted sklearn.linear_model.LogisticRegression with intercept and large C
            x:     matrix on which the model was fit
        This function uses asymtptics for maximum likelihood estimates.
        """
        p = model.predict_proba(x)
        n = len(p)
        m = len(model.coef_[0]) + 1
        coefs = np.concatenate([model.intercept_, model.coef_[0]])
        x_full = np.matrix(np.insert(np.array(x), 0, 1, axis=1))
        ans = np.zeros((m, m))

        for i in range(n):
            ans = ans + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p[i, 1] * p[i, 0]

        vcov = np.linalg.inv(np.matrix(ans))
        se = np.sqrt(np.diag(vcov))

        t = coefs / se  # t values
        p = (1 - norm.cdf(abs(t))) * 2

        return p


                                                # # # # # # # # # # #
                                                # GENERATE  RESULTS #
                                                # # # # # # # # # # #


    def initialize(self, path, suma, go, ge):

        self.set_basics(dm_path=path, substitution_matrix=suma)
        self.gap_open = go
        self.gap_extend = ge
        self.set_graph_v2()
        self.tril_to_full_dm()
        self.set_embedding(0.4)
        self.cluster_vector = [1 for _ in range(self.number_of_sequences)]
        self.plot_cluster(self.cluster_vector, 'UMAP projection of TCR sequences using ' + self.substitution_matrix + ' ' + str(self.gap_open) + '/' + str(self.gap_extend))

    def run(self, gammas, c=1, independent=False):

        self.find_cluster(gammas[0])
        self.calculate_feature_vector()
        fv = self.feature_vector
        sv = self.sequences_per_cluster
        cv = [self.cluster_vector]

        for gamma in gammas[1:]:
            self.find_cluster(gamma)
            self.calculate_feature_vector()
            fv = np.concatenate((fv, self.feature_vector), axis=1)
            sv.extend(self.sequences_per_cluster)

            cv_inc = len(np.unique(cv))
            cv_append = [x+cv_inc for x in self.cluster_vector]
            cv.append(cv_append)
            self.cv_idxs.append(cv_inc)

        self.feature_vector = fv
        self.sequences_per_cluster = sv
        self.cluster_vector = cv

        self.set_response()
        if independent:
            self.split_feature_independent()
        else:
            self.split_feature()

        self.sk_build_model(reg_c=c)

        self.print_confusion_matrix()
        self.plot_roc()
        self.model_evaluation()
        self.logit_pvalue(self.result, self.train_x)

    def cv_reducer_run(self):
        self.feature_selection_cv(scoring='balanced_accuracy', plot=True)
        self.feature_vector = self.reduced_feature_vector
        self.split_feature()
        self.sk_build_model()

        self.print_confusion_matrix()
        self.plot_roc()
        self.model_evaluation()

    def man_reducer_run(self, upper):
        models = dict()
        for i in range(2, upper):
            rfe = RFE(estimator=LogisticRegression(max_iter=500000), n_features_to_select=i)
            model = LogisticRegression(max_iter=500000)
            models[str(i)] = Pipeline(steps=[('s', rfe), ('m', model)])

        results, names = [], []
        for name, model in models.items():
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
            scores = cross_val_score(model, self.feature_vector, self.response, scoring='balanced_accuracy', cv=cv, n_jobs=-1, error_score='raise')

            results.append(scores)
            names.append(name)
            print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))

        plt.boxplot(results, labels=names, showmeans=True)
        plt.ylabel('Cross validation score (balanced accuracy)')
        plt.xlabel('Number of features selected')
        plt.title('Feature selection ' + self.substitution_matrix + ' ' + str(self.gap_open) + '/' + str(self.gap_extend))
        plt.show()

        self.rfe_models = models

    def man_reducer_select(self, ix):
        ix = str(ix)
        rfe = self.rfe_models[ix][0]
        rfe.fit(self.feature_vector, self.response)
        mask = rfe.get_support(indices=True)
        self.feature_vector = rfe.transform(self.feature_vector)
        self.reduced_sequences = [self.reduced_sequences[ix] for ix in mask]
        self.tracing = [self.tracing[ix] for ix in mask]


    def sample_significant_feature(self):
        p_vals = []
        for train_index, test_index in self.cv.split(self.feature_vector, self.response):
            self.response = np.array(self.response)
            self.train_x, _ = self.feature_vector[train_index], self.feature_vector[test_index]
            self.train_y, _ = self.response[train_index], self.response[test_index]
            self.sk_build_model()
            p_vals.append(list(self.logit_pvalue(self.result, self.train_x))[1:])

        p_vals = list(zip(*p_vals))

        beta = LogisticRegressionCV(cv=self.cv, solver='liblinear', penalty='l1', Cs=100000000).fit(self.feature_vector, self.response).coef_[0]
        sequences = self.reduced_sequences

        hmp = []
        for i in range(len(p_vals)):
            hmp.append(1/sum([0.2/p for p in p_vals[i]]))

        beta_p = list(zip(hmp, beta, sequences))

        self.positive_significant_feature = []
        self.negative_significant_feature = []
        self.positive_tracing = []
        self.negative_tracing = []

        for ix, (p, feature, seq) in enumerate(beta_p):
            if p < 0.05:
                if feature > 0:
                    self.positive_significant_feature.append(seq)
                    self.positive_tracing.append(self.tracing[ix])
                else:
                    self.negative_significant_feature.append(seq)
                    self.negative_tracing.append(self.tracing[ix])

    def eliminate_colinear_features(self):
        corr = np.corrcoef(self.feature_vector, rowvar=0)
        w,v = np.linalg.eig(corr)

        n, m = self.feature_vector.sort()

        for ix, eigen_value in enumerate(w):
            if eigen_value < 0.01:
                self.feature_vector = np.delete(self.feature_vector, ix, axis=1)

        j, k = self.feature_vector.shape
        print(str(m-k), ' features deleted')

    def stability_run(self, gammas, c=1, independent=False):
        # Initial Louvain
        self.find_cluster(gammas[0])
        self.calculate_feature_vector()
        fv = self.feature_vector
        sv = self.sequences_per_cluster
        cv = [self.cluster_vector]

        # Concatenate Features
        for gamma in gammas[1:]:
            self.find_cluster(gamma)
            self.calculate_feature_vector()
            fv = np.concatenate((fv, self.feature_vector), axis=1)
            sv.extend(self.sequences_per_cluster)
            cv_inc = len(np.unique(cv))
            cv_append = [x+cv_inc for x in self.cluster_vector]
            cv.append(cv_append)
            self.cv_idxs.append(cv_inc)

        # Results from Concatenation
        self.feature_vector = fv
        self.sequences_per_cluster = sv
        self.cluster_vector = cv

        # Eliminate colinear Features
        self.tracing = []
        corr = np.corrcoef(self.feature_vector, rowvar=0)
        w, v = np.linalg.eig(corr)
        n, m = self.feature_vector.shape
        del_ix = []
        for ix, eigen_value in enumerate(w):
            if eigen_value < 0.01:
                del_ix.append(ix)
        j, k = self.feature_vector.shape
        self.feature_vector = np.delete(self.feature_vector, del_ix, axis=1)

        reduced_sequences = []
        for ix in del_ix:  # loop through slected features indexes

            reduced_sequences.append(self.sequences_per_cluster[ix])

            for run_ix, run in enumerate(self.cluster_vector):
                # loop through every clustering
                if ix in run:  # if selected cluster in this run
                    temp = []  # store selected feature traceback here

                    for s_ix, s in enumerate(run):
                        # correct cv to a format which enables tracback
                        if s == ix:
                            temp.append(0)
                        else:
                            temp.append(1)
                    self.tracing.append(temp)
                    break
                else:
                    continue
        sequences = self.reduced_sequences
        print(str(m - k), ' features deleted')

        # Here stability selection is instantiated and run
        selector = StabilitySelection(base_estimator=LogisticRegression(penalty='liblinear'), lambda_name='C',
                                      lambda_grid=np.logspace(-5, -1, 50), bootstrap_func='stratified', n_jobs=-1).fit(self.feature_vector, self.response)

        idxs = selector.get_support(indices=True)
        self.feature_vector = selector.transform(self.feature_vector)

        reduced_sequences = []
        self.tracing = []  # store every traceback vector here
        for ix in idxs:  # loop through slected features indexes

            reduced_sequences.append(self.sequences_per_cluster[ix])

            for run_ix, run in enumerate(self.cluster_vector):
                # loop through every clustering
                if ix in run:  # if selected cluster in this run
                    temp = []  # store selected feature traceback here

                    for s_ix, s in enumerate(run):
                        # correct cv to a format which enables tracback
                        if s == ix:
                            temp.append(0)
                        else:
                            temp.append(1)
                    self.tracing.append(temp)
                    break
                else:
                    continue
        sequences = self.reduced_sequences

        # Model Evaluation
        self.model = LogisticRegression(solver='liblinear', penalty='l1')

        cv_f1_score = np.average(np.array(cross_val_score(self.model, self.feature_vector, self.response, cv=5, scoring='f1')))
        cv_rocauc_score = np.average(np.array(cross_val_score(self.model, self.feature_vector, self.response, cv=5, scoring='roc_auc')))
        cv_prec_score = np.average(np.array(cross_val_score(self.model, self.feature_vector, self.response, cv=5, scoring='precision')))
        cv_balanced_acc_score = np.average(np.array(cross_val_score(self.model, self.feature_vector, self.response, cv=5, scoring='balanced_accuracy')))

        cv_mcc = []
        cv_log_loss = []

        self.cv = StratifiedKFold(n_splits=5)
        for train_index, test_index in self.cv.split(self.feature_vector, self.response):
            self.response = np.array(self.response)
            self.train_x, self.test_x = self.feature_vector[train_index], self.feature_vector[test_index]
            self.train_y, self.test_y = self.response[train_index], self.response[test_index]
            self.model.fit(self.train_x, self.train_y)

            y_pred = self.model.predict(self.test_x)
            mcc = matthews_corrcoef(self.test_y, y_pred)
            cv_mcc.append(mcc)

            y_pred_proba = self.model.predict_proba(self.test_x)
            ll = log_loss(self.test_y, y_pred_proba)
            cv_log_loss.append(ll)

        mcc = np.average(np.array(cv_mcc))
        ll = np.average(np.array(cv_log_loss))

        print('MCC Score: \t %.3f' % mcc)
        print('Log-Loss: \t %.3f' % ll)
        print('CV F1 Score: \t %.3f' % cv_f1_score)
        print('CV ROCAUC Score: \t %.3f' % cv_rocauc_score)
        print('CV AVG PREC Score: \t %.3f' % cv_prec_score)
        print('CV AVG BALANCED ACC Score: \t %.3f' % cv_balanced_acc_score)

        # P Values
        p_vals = []
        for train_index, test_index in self.cv.split(self.feature_vector, self.response):
            self.response = np.array(self.response)
            self.train_x, _ = self.feature_vector[train_index], self.feature_vector[test_index]
            self.train_y, _ = self.response[train_index], self.response[test_index]
            self.result = self.model.fit(self.train_x, self.train_y)

            p = self.result.predict_proba(self.train_x)
            n = len(p)
            m = len(self.result.coef_[0]) + 1
            coefs = np.concatenate([self.result.intercept_, self.result.coef_[0]])
            x_full = np.matrix(np.insert(np.array(self.train_x), 0, 1, axis=1))
            ans = np.zeros((m, m))

            for i in range(n):
                ans = ans + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p[i, 1] * p[i, 0]

            vcov = np.linalg.inv(np.matrix(ans))
            se = np.sqrt(np.diag(vcov))

            t = coefs / se  # t values
            p = (1 - norm.cdf(abs(t))) * 2

            p_vals.append(list(p)[1:])

        p_vals = list(zip(*p_vals))

        self.result = LogisticRegressionCV(cv=self.cv, penalty='l1', solver='liblinear', max_iter=500000).fit(self.feature_vector, self.response)
        beta = self.result.coef_[0]

        hmp = []
        for i in range(len(p_vals)):
            hmp.append(1 / sum([0.2 / p for p in p_vals[i]]))

        beta_p = list(zip(hmp, beta, sequences))

        self.positive_significant_feature = []
        self.negative_significant_feature = []
        self.positive_tracing = []
        self.negative_tracing = []

        for ix, (p, feature, seq) in enumerate(beta_p):
            if p < 0.05:
                if feature > 0:
                    self.positive_significant_feature.append(seq)
                    self.positive_tracing.append(self.tracing[ix])
                else:
                    self.negative_significant_feature.append(seq)
                    self.negative_tracing.append(self.tracing[ix])
