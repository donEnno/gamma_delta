# Default
import time

from joblib import dump, load, Parallel
import numexpr
import numpy as np
import pandas as pd
import os

# UMAP dependencies
import seaborn as sns

# Dimension reduction and clustering
import umap
import umap.plot
import networkit
import networkx as nx
from community import community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# To avoid
# INFO:numexpr.utils:Note: NumExpr detected 28 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 8.
# INFO:numexpr.utils:NumExpr defaulting to 8 threads.
numexpr.MAX_THREADS = 14

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                            RESOLUTION = 0.875                          GAMMA = 1.05                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_umap(df: pd.DataFrame):
    """
    Reduces the input data into two-dimensional space using the UMAP method.

    :param df: pandas df
    :return Reduced data np.ndarray
    """
    sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(df)        # type(embedding) = <class 'numpy.ndarray'

    return embedding


def numpy_to_nk_graph(dist_mat: np.ndarray):
    """
    Converts a distance matrix to a NetworKit graph object.

    :param dist_mat: Input-distance matrix,
    :return: NetworKit graph of dist_mat
    """
    m, _ = dist_mat.shape
    g = networkit.Graph(m, weighted=True)

    mask_x, mask_y = np.mask_indices(m, np.tril, -1)
    masking_zip = zip(mask_x, mask_y, dist_mat[mask_x, mask_y])

    for nodeA, nodeB, weight in masking_zip:
        g.addEdge(nodeA, nodeB, weight)

    return g


def calculate_partitions(patient: str, substitution_matrix: str, gamma=1.00, save_partition=False, save_plot=False,
                         show=False):
    """
    Plots the result of the Louvain community detection algorithm in a UMAP. One can either you use a networkx-based
    version of the algorithm or a NetworKit-based version. \n
    It is optional to either save the plot or the partitions found.

    :param patient: Desired samples: 'BL', 'HD', 'FU' (or any combination [ IN PROGRESS ])
    :param substitution_matrix: Desired substitution matrix, all caps
    :param gamma: Resolution in NetworKit approach
    :param save_partition:Toggle for saving partition to file
    :param save_plot: Toggle for saving plot to file
    :param show: Toggle for plt.show()
    :returns None
    """
    # TODO Adjust for new file formats.
    if patient == 0:
        pat = f'ALL_SEQUENCES_{substitution_matrix}_DISTANCE_MATRIX'
    else:
        pat = fr'PATIENT_{patient}_{substitution_matrix}_DISTANCE_MATRIX'

    # TODO '/home/ubuntu/Enno/gammaDelta/distance_matrices/{pat}'
    patient_distance_matrix = load(rf"/home/ubuntu/Enno/mnt/volume/distance_matrices/TEST")

    ix, iy = patient_distance_matrix.shape

    patient_df = pd.DataFrame(data=patient_distance_matrix, index=[f'Sequence_{i}' for i in range(1, ix + 1)],
                              columns=[f'Sequence_{i}' for i in range(1, iy + 1)])

    # CREATE UMAP
    embedding = get_umap(patient_df)

    # CREATE GRAPH
    start_time = time.time()
    g = numpy_to_nk_graph(patient_distance_matrix)
    end_time = time.time()
    print('Creating the networKit graph took', round(end_time-start_time, 2))

    # DETECT COMMUNITIES
    print('Detecting communities using gamma = ', gamma)
    partition = networkit.community.detectCommunities(g, algo=networkit.community.PLM(g, refine=True, gamma=gamma))
    if save_partition:
        subname = f'_gamma={gamma}_'
        dump(partition,
                    fr'/home/ubuntu/Enno/mnt/volume/partitions/patient_{patient}_{substitution_matrix}{subname}communities')

    # PLOT LOUVAIN CLUSTERING
    if save_plot:
        cmap = cm.get_cmap('prism', max(partition.getVector()) + 1)
        plt.scatter(embedding[:, 0], embedding[:, 1], cmap=cmap, c=list(partition.getVector()), s=5, alpha=0.5)
        sub = f' gamma {gamma}'
        # TODO {substitution_matrix}
        plt.title(f'Louvain com. det. in UMAP projection of all patients using identity with {sub}', fontsize=15)
        # TODO PATIENT_{patient}_{substitution_matrix}_nk_{gamma*1000}
        plt.savefig(fr'/home/ubuntu/Enno/gammaDelta/plots/TEST.png')
        plt.clf()
    if show:
        plt.show()
    return partition


if __name__ == '__main__':
    mat = load(rf"/home/ubuntu/Enno/mnt/volume/distance_matrices/NP_BLHD_DM")

    ix, iy = patient_distance_matrix.shape
    patient_df = pd.DataFrame(data=patient_distance_matrix, index=[f'Sequence_{i}' for i in range(1, ix + 1)],
                             columns=[f'Sequence_{i}' for i in range(1, iy + 1)])
    print(len(patient_df))
    g = numpy_to_nk_graph(patient_distance_matrix)

    print(g.numberOfEdges())
    print(g.numberOfNodes())

    """ print('Let\'s go!')
        calculate_partitions(0, 'ClustalO', netx=False, gamma=1.06, save_plot=True, show=True)
    """