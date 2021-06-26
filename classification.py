# Default
import os

import joblib
from matplotlib import pyplot, pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# Pipeline
from partitions import get_frequencies
from projection import get_umap, load_dm

# ML
from sklearn.linear_model import LogisticRegression     # , LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split    # , StratifiedShuffleSplit
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# Generate feature
partition = joblib.load(fr"/home/ubuntu/Enno/mnt/volume/vectors/blhd_ID_%.2f_communities" % 1.00)
x, aa_cluster = get_frequencies(partition, ['BL', 'HD'], absolute_toggle=True)

# Set response
bl = [1 for i in range(66)]
fu = [1 for j in range(55)]
hd = [0 for k in range(29)]
y = []
y.extend(bl)
# y.extend(fu)
y.extend(hd)


def eval_model(model, testX, testY, c, roc=False, write_cluster_to_fasta=False):
    """
    :param model: Model to be evaluated.
    :param testX: Test features.
    :param testY: Test responses.
    :param c: Regularization parameter. For low values a stronger regularization is implied.
    :param roc: Set True if ROC-Curve should be plotted.
    """

    # Model score
    print(11*'= ', ' MODEL SCORES ', 11*' =', '\n')

    test_score = model.score(testX, testY)
    print('Model score for testX/Y:%.3f' % test_score)
    total_score = model.score(x, y)
    print('Model score for x/y: %.3f' % total_score)

    # Predict probabilities
    lr_probs = model.predict_proba(testX)
    lr_probs = lr_probs[:, 1]                   # positive outcomes only

    # Calculate scores
    lr_auc = roc_auc_score(testY, lr_probs)
    print('Logistic: ROC AUC=%.3f' % lr_auc)

    # Report
    c_report = classification_report(testY, model.predict(testX))
    print(c_report)

    if roc:
        # Calculate roc curves
        lr_fpr, lr_tpr, _ = roc_curve(testY, lr_probs)

        # Plot the curve for the model
        pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
        pyplot.title(label=f'Logistic: ROC AUC=%.3f - C={c}' % lr_auc)
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.legend()
        pyplot.show()
        pyplot.clf()

    # Print confusion matrix to Run.
    co_ma = confusion_matrix(testY, model.predict(testX))
    print(r'A\P', '\t', '0', '\t', '1')
    print(12*'=')

    for i in range(2):
        print(i, ' \t', co_ma[i, 0], '\t', co_ma[i, 1])
        print(12 * '=')

    # Coef
    coef = model.coef_[0]
    # Zip one clusters coefficient, its SeqIO objects and its index.
    coef_cluster = zip(coef, aa_cluster, range(len(coef)))
    coef_cluster = list(coef_cluster)

    # TODO Get it right.
    res = sorted(coef_cluster, key=lambda cf: cf[0])
    print(res[0][0], res[0][-1])
    print(res[-1][0], res[-1][-1])

    if write_cluster_to_fasta:
        # highest_coef_cluster = []
        with open('/home/ubuntu/Enno/gammaDelta/sequence_data/BLHD_LOWEST_COEF.fasta', 'w') as wf:
            for ele in res[-1][1]:
                content = '>' + ele.id + '\n' + ele.seq + '\n'
                wf.writelines(content)


def adjust_partition(partition):
    """
    Sometimes the enumeration is messed up and this function maps them again properly.
    """
    num_partitions = len(np.unique(partition))
    if max(partition) != num_partitions:
        dic = dict(zip(np.unique(partition), range(num_partitions)))
        partition = [dic[i] for i in partition]

        return partition


def prepare_plot_data(cluster_of_interest):
    z = []

    for cluster in partition:
        if cluster == cluster_of_interest:
            z.append(1)
        else:
            z.append(0)
    z = np.array(z)
    return z

def highlight_cluster_in_umap(distance_matrix, cluster_of_interest):
    """
    Highlights the cluster of interest in the UMAP projection.

    :param distance_matrix: Distance matrix to be used.
    """

    # TODO Make generic.
    adjust_partition(partition)
    print(partition)

    # CREATE UMAP
    ix, iy = distance_matrix.shape
    patient_df = pd.DataFrame(data=distance_matrix, index=[f'Sequence_{i}' for i in range(1, ix + 1)],
                              columns=[f'Sequence_{i}' for i in range(1, iy + 1)])
    # Data to be plotted
    # TODO Take close look at embedding.
    embedding = get_umap(patient_df)
    vector = prepare_plot_data(cluster_of_interest)

    cmap = mcolors.ListedColormap(["grey", "red"])
    plt.scatter(embedding[:, 0], embedding[:, 1], c=vector, cmap=cmap, s=5, alpha=0.25)
    plt.title(f'Found clusters', fontsize=15)
    plt.show()
    plt.clf()


if __name__ == '__main__':

    do_modeling = True
    if do_modeling:
        for c in [0.01]:
            print(10 * '= ', 'RUNNING ON C =', c, 10 * ' =')

            # Split into train/test sets
            trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

            # Fit model
            model = LogisticRegression(solver='lbfgs', n_jobs=-1, C=c, random_state=3, max_iter=5000)
            model.fit(trainX, trainY)

            eval_model(model, testX, testY, c)

    # print(number_of_communities)
    # [(1.00,  3) 1 2, (1.01,  6), (1.02,  7), (1.03, 10), (1.04, 16), (1.05, 19),
    #  (1.06, 29), (1.07, 43), (1.08, 59), (1.09,  8), (1.1,   9), (1.11, 12),
    #  (1.12, 17), (1.13, 24), (1.14, 28), (1.15, 39), (1.16, 64), (1.17, 111), (1.18, 301)]

    #dm = load_dm("/home/ubuntu/Enno/mnt/volume/distance_matrices/TEST")
    #highlight_cluster_in_umap(dm, 1)
    #highlight_cluster_in_umap(dm, 2)

"""
    path = '/home/ubuntu/Enno/mnt/volume/vectors/'
    list_of_files = os.listdir(path)
    list_of_files = [x for x in list_of_files if not x.startswith('c')]
    list_of_files = sorted(list_of_files)
    list_of_files = [(x, float(x[8:12])) for x in list_of_files]
    print(list_of_files)

    number_of_communities = []
    for ele in list_of_files:

        com = joblib.load(path+ele[0])
        com = len(np.unique(com))

        number_of_communities.append((ele[1], com))
"""
