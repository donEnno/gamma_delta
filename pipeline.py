# Imports
# Defaults
import os
import errno
import time
from joblib import load
import pandas as pd
import numpy as np
import sys
from itertools import combinations, product

# Alignments
from Bio import SeqIO

# Projection
import seaborn as sns
import umap
import umap.plot
import networkit
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ML
import statsmodels.api as sm
from scipy.stats import norm

from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import l1_min_c
from sklearn.linear_model import LogisticRegression     # , LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix,  accuracy_score, roc_curve, roc_auc_score, \
    balanced_accuracy_score, log_loss, brier_score_loss, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2, RFECV
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.estimator_checks import check_estimator


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

    def set_embedding(self): # , spread, min_dist, a, b
        sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})
        reducer = umap.UMAP() # spread=spread, min_dist=min_dist, a=a, b=b
        self.embedding = reducer.fit_transform(self.dm)  # type(embedding) = <class 'numpy.ndarray'

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

    def find_cluster(self, gamma):

        cluster = networkit.community.detectCommunities(self.graph,
                                                        algo=networkit.community.PLM(self.graph, refine=True, gamma=gamma))
        cluster_vector = cluster.getVector()
        self.cluster = cluster
        self.cluster_vector = cluster_vector
        self.number_of_clusters = len(np.unique(self.cluster_vector))

    def plot_cluster(self):

        cmap = cm.get_cmap('prism', max(self.cluster_vector) + 1)
        x = self.embedding[:, 0]
        y = self.embedding[:, 1]
        plt.scatter(x, y, cmap=cmap, c=list(self.cluster_vector), s=5, alpha=0.5)
        plt.title(f'Louvain com. det. in UMAP projection using gamma ' + str(self.gamma) +
                  '\n' + str(self.number_of_clusters) + 'cluster found', fontsize=15)
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
            # return np.array(absolute), sequences_per_cluster
            self.feature_vector = np.array(absolute)
        else:
            # return np.array(relative), sequences_per_cluster
            self.feature_vector = np.array(relative)
        self.sequences_per_cluster.append(sequences_per_cluster)

    def concatenate_features(self, gammas):
        con_feature = []

        for gamma in gammas:
            self.find_cluster(gamma=gamma)
            self.calculate_feature_vector()

            print(self.feature_vector.shape)
            print(self.feature_vector)

            con_feature.append(self.feature_vector)

        new_feature = np.concatenate(con_feature, axis=1)
        self.feature_vector = new_feature

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

    def feature_selection(self, scoring, plot=False):
        min_features_to_select = 1

        rfecv = RFECV(estimator=self.model, step=1, cv=StratifiedKFold(5), scoring=scoring,
                      min_features_to_select=min_features_to_select)
        rfecv.fit(self.feature_vector, self.response)
        self.reduced_feature_vector = rfecv.transform(self.feature_vector)

        if plot:
            plt.figure()
            plt.xlabel("Number of features selected")
            plt.ylabel("Cross validation score (nb of correct classifications)")
            plt.plot(range(min_features_to_select,
                           len(rfecv.grid_scores_) + min_features_to_select),
                     rfecv.grid_scores_)
            plt.show()

    def sk_build_model(self, l1_ratio=None, penalty='l2', solver='lbfgs', reg_c=1):
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

    def sm_build_model(self, solver='lbfgs', alpha=0, reg=False):
        """

        :param solver: ‘newton’ for Newton - Raphson, ‘nm’ for Nelder-Mead
                ‘bfgs’ for Broyden - Fletcher - Goldfarb - Shanno(BFGS)
                ‘lbfgs’ for limited - memory BFGS with optional box constraints
                ‘powell’ for modified Powell’s method
                ‘cg’ for conjugate gradient
                ‘ncg’ for Newton - conjugate gradient
                ‘basinhopping’ for global basin-hopping solver
        :return:
        """
        self.model = sm.Logit(self.train_y, self.train_x)

        if reg:
            self.result = self.model.fit_regularized(method='l1', alpha=alpha, maxiter=50000)
        else:
            self.result = self.model.fit(method=solver, maxiter=50000)

        print(self.result.summary())

    def prep_grid_search(self, gamma):
        # TODO
        self.find
        model = LogisticRegression()
    # # # # # # # # # # # # # # # #
    #   MODEL EVALUATION METHODS  #
    # # # # # # # # # # # # # # # #

    def sk_model_evaluation(self):
        self.model_score = self.result.score(self.test_x, self.test_y)

    def sk_evaluate_model(self):
        self.model_report = classification_report(self.test_y, self.result.predict(self.test_x))
        # TODO y_pred[:, 1] ?
        y_pred = self.model.predict_proba(self.test_x)[:, 1]
        ll = log_loss(self.test_y, y_pred)
        print(self.model_report)
        print('Log loss of current model is %d' % ll)

    def plot_calibration_curve(self, fig_index, name='Log. Reg.'):
        """Plot calibration curve for est w/o and with calibration. """

        # Logistic regression with no calibration as baseline
        lr = self.model

        fig = plt.figure(fig_index, figsize=(10, 10))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))

        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

        lr.fit(self.train_x, self.train_y)
        y_pred = lr.predict(self.test_x)
        if hasattr(lr, "predict_proba"):
            prob_pos = lr.predict_proba(self.test_x)[:, 1]
        else:  # use decision function
            prob_pos = lr.decision_function(self.test_x)
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        lr_score = brier_score_loss(self.test_y, prob_pos, pos_label=1)
        print("%s:" % name)
        print("\tBrier: %1.3f" % lr_score)
        print("\tPrecision: %1.3f" % precision_score(self.test_y, y_pred))
        print("\tRecall: %1.3f" % recall_score(self.test_y, y_pred))
        print("\tF1: %1.3f\n" % f1_score(self.test_y, y_pred))

        fraction_of_positives, mean_predicted_value = calibration_curve(self.test_y, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, lr_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title('Calibration plots  (reliability curve)')

        ax2.set_xlabel("Mean predicted value")
        ax2.set_ylabel("Count")
        ax2.legend(loc="upper center", ncol=2)

        plt.tight_layout()
        plt.show()

    def sk_cv_scores(self):
        self.cv_f1_score = np.average(np.array(cross_val_score(self.model, self.feature_vector, self.response, cv=5, scoring='f1')))
        self.cv_rocauc_score = np.average(np.array(cross_val_score(self.model, self.feature_vector, self.response, cv=5, scoring='roc_auc')))
        self.cv_prec_score = np.average(np.array(cross_val_score(self.model, self.feature_vector, self.response, cv=5, scoring='precision')))
        self.cv_balanced_acc_score = np.average(np.array(cross_val_score(self.model, self.feature_vector, self.response, cv=5, scoring='balanced_accuracy')))

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
        print(r'A\P', '\n', '\t', '0', '\t', '1')
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
        t = coefs / se
        p = (1 - norm.cdf(abs(t))) * 2

        return p, t

    def sm_feature_plot(self):

        proba = 1 / (1 + np.exp(-self.result.fittedvalues))
        inc = 0.1
        _ = plt.hist(proba, bins=[i*inc for i in range(11)])
        plt.title('Hist with 10 bins')
        plt.show()

    def regularization_path(self):
        cs = l1_min_c(self.feature_vector, self.response, loss='log') * np.logspace(0, 7, 16)

        print("Computing regularization path ...")
        start = time()
        clf = LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=int(1e6), warm_start=True,
                                 intercept_scaling=10000.)
        coefs_ = []
        for c in cs:
            clf.set_params(C=c)
            clf.fit(self.feature_vector, self.response)
            coefs_.append(clf.coef_.ravel().copy())
        print("This took %0.3fs" % (time() - start))

        coefs_ = np.array(coefs_)
        plt.plot(np.log10(cs), coefs_, marker='o')
        ymin, ymax = plt.ylim()
        plt.xlabel('log(C)')
        plt.ylabel('Coefficients')
        plt.title('Logistic Regression Path')
        plt.axis('tight')
        plt.show()

    def performance_sampling(self):

        model = self.result
        significant_cluster = len([1 for pvalue in model.pvalues if pvalue < 0.05])

        content = [self.gamma, self.number_of_clusters, significant_cluster, model.llr_pvalue, self.cv_f1_score,
                   self.cv_rocauc_score, self.cv_prec_score, self.cv_balanced_acc_score]

        self.performance.append(content)

    def plot_performance(self, sm):

        df = pd.DataFrame(self.performance,
                          columns=['gamma', 'cluster', 'significant_cluster', 'model_pvalue', 'f1', 'rocauc', 'prec',
                                   'bal_acc'])

        plt.scatter(df['cluster'], df['f1'], label='f1')
        plt.scatter(df['cluster'], df['rocauc'], label='rocauc')
        plt.scatter(df['cluster'], df['prec'], label='prec')
        plt.scatter(df['cluster'], df['bal_acc'], label='bal_acc')
        # plt.plot([i for i in range(len(df['cluster']))], [0.5 for i in range(len(df['f1']))], 'k--')
        plt.legend()
        plt.title('Performance vs. number of cluster for %s' % sm)
        plt.show()
        plt.clf()

    def append_cluster_of_interest(self):

        coef = list(zip(self.result.params, self.result.pvalues))
        coef = list(enumerate(coef))

        threshold = 0.05

        if self.result.llr_pvalue < threshold: # model p-value < 0.05
            for i, (b, p) in coef:
                if p < threshold: # coef p value < 0.05
                    if b < 0:
                        self.cluster_of_interest_zero.append([i, (b, p), self.sequences_per_cluster[-1][i]])
                    elif b > 0:
                        self.cluster_of_interest_one.append([i, (b, p), self.sequences_per_cluster[-1][i]])
                    else:
                        print('Cluster has coefficient zero')

    def compute_cluster_overlap(self):
        num_of_coi_one = len(self.cluster_of_interest_one)
        num_of_coi_zero = len(self.cluster_of_interest_zero)

        print('#COI one:', num_of_coi_one)
        print('#COI zero:', num_of_coi_zero)

        list_of_overlaps_one = []
        list_of_overlaps_zero = []

        for i, (b1, p1), s1 in self.cluster_of_interest_one:
            for j, (b2, p2), s2 in self.cluster_of_interest_one:
                print(type(s1), len(s1))
                print(type(s2), len(s2))
                if s1 == s2:
                    print('ID worked.')
                else:
                    try:
                        seq1 = [seq.seq for i, seq in s1]
                        seq2 = [seq.seq for i, seq in s2]
                        overlap = [seq for seq in seq1 if seq in seq2]
                        print('%s sequences overlap, that equals a fraction of %s' % (str(len(overlap)), str(len(overlap)/max(len(s1), len(s2)))))
                        list_of_overlaps_one.append(overlap)
                    except NotImplementedError:
                        print('NotImplementedError avoided.')
                        pass

        for i, (b1, p1), s1 in self.cluster_of_interest_zero:
            for j, (b2, p2), s2 in self.cluster_of_interest_zero:
                print(type(s1), len(s1))
                print(type(s2), len(s2))
                if s1 == s2:
                    print('ID worked.')
                else:
                    try:
                        seq1 = [seq.seq for i, seq in s1]
                        seq2 = [seq.seq for i, seq in s2]
                        overlap = [seq for seq in seq1 if seq in seq2]
                        print('%s sequences overlap, that equals a fraction of %s' % (str(len(overlap)), str(len(overlap) / max(len(s1), len(s2)))))
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

        fig, ax = plt.subplots(figsize=(40, 16))
        labels = [str(i) for i in range(1, 96)]

        first = True
        bottom = [0 for _ in range(1, 96)]
        width = 0.7
        for i in range(num_of_cluster):
            if first:
                first = False
                ax.bar(labels, temp[i], width=width, label='Cluster %i' % (i + 1), bottom=bottom)
                bottom = temp[i]
            else:
                ax.bar(labels, temp[i], width=width, label='Cluster %i' % (i + 1), bottom=bottom)
                bottom = [i + j for i, j in zip(bottom, temp[i])]

        ax.set_ylabel('Frequency', fontsize=24)
        ax.set_title('Distribution of BL and HD sequences among cluster ', fontsize=24)
        ax.legend(fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()
        plt.clf()

                     # maxiter=,
                     # method=
                     # ‘newton’ for Newton - Raphson, ‘nm’ for Nelder-Mead
                     # ‘bfgs’ for Broyden - Fletcher - Goldfarb - Shanno(BFGS)
                     # ‘lbfgs’ for limited - memory BFGS with optional box constraints
                     # ‘powell’ for modified Powell’s method
                     # ‘cg’ for conjugate gradient
                     # ‘ncg’ for Newton - conjugate gradient
                     # ‘basinhopping’ for global basin-hopping solver
                     # ‘minimize’ for generic wrapper of scipy minimize (BFGS by default)

                     # fit_regularized([start_params, method, …]) - - - Fit the model using a regularized maximum likelihood.

