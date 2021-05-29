# Default
import os.path
import time

import joblib
import networkit
import numexpr

from distances import get_num_seq


# UMAP dependencies
import numpy as np
import pandas as pd
import seaborn as sns
import datashader
import bokeh
import holoviews
import colorcet

# Dimension reduction and clustering
import umap
import umap.plot
import matplotlib.pyplot as plt
from community import community_louvain
import matplotlib.cm as cm
import networkx as nx
import sklearn.cluster as cluster
from networkit import *

# To avoid
# INFO:numexpr.utils:Note: NumExpr detected 28 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 8.
# INFO:numexpr.utils:NumExpr defaulting to 8 threads.
numexpr.MAX_THREADS = 14


def get_umap(df, patient, substitution_matrix, show=False):
    """
    :param df: pandas df
    :param patient: int, used #patient
    :param substitution_matrix: used substitution matrix,
    :param show: boolean, toggle UMAP projection plot
    :return numpy.ndarray
    """
    sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(df)        # type(embedding) = <class 'numpy.ndarray'

    if show:
        plt.scatter(embedding[:, 0], embedding[:, 1])
        plt.title(f'UMAP projection of Patient {patient} using {substitution_matrix}')
        plt.show()
        plt.close()

    return embedding


def numpy_to_nk_graph(dist_mat):
    """
    :param dist_mat: numpy array
    :return: NetworKit graph of dist_mat
    """
    m, _ = dist_mat.shape
    G = networkit.Graph(m, weighted=True)

    mask_x, mask_y = np.mask_indices(m, np.tril, -1)
    masking_zip = zip(mask_x, mask_y, dist_mat[mask_x, mask_y])

    for nodeA, nodeB, weight in masking_zip:
        G.addEdge(nodeA, nodeB, weight)

    return G


def numpy_to_nx_graph(dist_mat):
    """
    :param dist_mat: numpy array
    :return: networkx graph of dist_mat
    """
    g = nx.Graph(weighted=True)

    for ix, row in enumerate(dist_mat):
        for iy, i in enumerate(row):
            g.add_edge(ix, iy, weight=i)
    return g


def plot_louvain(patient, substitution_matrix, n, resolution=1.0, gamma=1.0, save_partition=False, save_plot=False, show=False):
    """
    :param patient: int, desired patient, 0 for all patients
    :param substitution_matrix: string, desired substitution matrix, all caps
    :param n: boolean, True for networkx-based Louvain algo, False for NetworKit-based algorithm
    :param resolution: float, resolution in networkx approach
    :param gamma: float, resolution in NetworKit approach
    :param save_partition: boolean, toggle for saving partition to file
    :param save_plot: boolean, toggle for saving plot to file
    :param show: boolean, toggle for plt.show()
    """

    if patient == 0:
        pat = f'ALL_SEQUENCES_{substitution_matrix}_DISTANCE_MATRIX'
    else:
        pat = fr'PATIENT_{patient}_{substitution_matrix}_DISTANCE_MATRIX'

    patient_distance_matrix = joblib.load(rf'/home/ubuntu/Enno/gammaDelta/distance_matrices/{pat}')
    ix, iy = patient_distance_matrix.shape

    patient_df = pd.DataFrame(data=patient_distance_matrix, index=[f'Sequence_{i}' for i in range(1, ix + 1)],
                              columns=[f'Sequence_{i}' for i in range(1, iy + 1)])

    # CREATE UMAP
    embedding = get_umap(patient_df, patient=patient, substitution_matrix=substitution_matrix)

    # CREATE GRAPH
    if n:
        G = numpy_to_nx_graph(patient_distance_matrix)
    else:
        G = numpy_to_nk_graph(patient_distance_matrix)

    # DETECT COMMUNITIES
    if n:
        partition = community_louvain.best_partition(G, resolution=resolution)
    else:
        partition = networkit.community.detectCommunities(G, algo=networkit.community.PLM(G, refine=True, gamma=gamma))
    if save_partition:
        if n: subname = f'_resolution={resolution}_'
        else: subname = f'_gamma={gamma}_'
        joblib.dump(partition,
                    fr'/home/ubuntu/Enno/gammaDelta/partition/patient_{patient}_{substitution_matrix}{subname}communities')

    # PLOT LOUVAIN CLUSTERING
    if n:
        cmap = cm.get_cmap('prism', max(partition.values()) + 1)
        plt.scatter(embedding[:, 0], embedding[:, 1], cmap=cmap, c=list(partition.values()), s=5)
        sub = f' resolution {resolution}'
    else:
        cmap = cm.get_cmap('prism', max(partition.getVector()) + 1)
        plt.scatter(embedding[:, 0], embedding[:, 1], cmap=cmap, c=list(partition.getVector()), s=5)
        sub = f' gamma {gamma}'
    if show:
        plt.title(f'Louvain com. det. in UMAP projection of all patients using {substitution_matrix} with {sub}', fontsize=15)
        if save_plot:
            plt.savefig(fr'/home/ubuntu/Enno/gammaDelta/plots/PATIENT_{patient}_{substitution_matrix}.png')
        plt.show()


if __name__ == '__main__':
    plot_louvain(0, 'BLOSUM45', n=False, gamma=1.05, show=True)


