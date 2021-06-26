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
#                           GAMMA = ~1.06 for NK approach
#                                 = ~
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def load_dm(path):
    dm = load(path)
    return dm


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

    # For sequence tracking purposes
    print('A')
    #l_invest = list(masking_zip)
    #print('B')
    #print(len(l_invest))
    textfile = open('masking_zip.txt', 'w')
    print('C')
    counter = 0
    for nodeA, nodeB, weight in masking_zip:                                                        # To investigate
        counter += 1
        if counter % 100000 == 0: print(counter)
        content = str(nodeA), ' ', str(nodeB), ' ', str(weight)
        textfile.write(''.join(content) + '\n')
    textfile.close()
    # end

    i = 0
    for nodeA, nodeB, weight in masking_zip:
        i += 1
        if i % 10 ** 8 == 0: print(i)
        g.addEdge(nodeA, nodeB, weight)
    print(i)
    return g


def calculate_partitions(patient: str, substitution_matrix: str, distance_matrix, graph, gamma=1.00, save_partition=False, save_plot=False,
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

    # CREATE UMAP
    ix, iy = distance_matrix.shape
    patient_df = pd.DataFrame(data=distance_matrix, index=[f'Sequence_{i}' for i in range(1, ix + 1)],
                              columns=[f'Sequence_{i}' for i in range(1, iy + 1)])
    embedding = get_umap(patient_df)

    # DETECT COMMUNITIES
    print('Detecting communities using gamma = ', gamma)
    partition = networkit.community.detectCommunities(graph, algo=networkit.community.PLM(graph, refine=True, gamma=gamma))
    vector = partition.getVector()
    num_partitions = len(np.unique(vector))

    if save_partition:
        dump(vector, fr'/home/ubuntu/Enno/mnt/volume/vectors/c_{substitution_matrix}_{gamma}_communities')

    # PLOT LOUVAIN CLUSTERING
    if save_plot:
        cmap = cm.get_cmap('prism', max(partition.getVector()) + 1)
        plt.scatter(embedding[:, 0], embedding[:, 1], cmap=cmap, c=list(vector), s=5, alpha=0.25)
        plt.title(f'Louvain com. det. in UMAP projection of all BLHD patients with {gamma} gamma \n'
                  f'Identity, {num_partitions} communites found', fontsize=15)
        plt.savefig(fr'/home/ubuntu/Enno/mnt/volume/plots/blhd_{substitution_matrix}_{gamma}g_cluster.png')
        if not show:
            plt.clf()
    if show:
        plt.show()
        plt.clf()
    return partition


if __name__ == '__main__':
    print('Let\'s go!')

    start_time = time.time()
    dm = load_dm(rf"/home/ubuntu/Enno/mnt/volume/distance_matrices/NP_BLHD_DM")
    end_time = time.time()
    print('Loading the DM took', round(end_time - start_time, 2), '[s]')

    start_time = time.time()
    g = numpy_to_nk_graph(dm)
    end_time = time.time()
    print('Creating the networKit graph took', round(end_time - start_time, 2), '[s]')

    calculate_partitions(0, 'CO', dm, g, gamma=1.01, save_partition=True)

