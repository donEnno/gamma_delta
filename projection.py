# Default
import os.path
import time

import joblib
import networkit
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


def plot_umap(patient, substitution_matrix, num_clusters=5, all_patients=False, plot=False, kmeans=False, louvain=False):
    """
    :param patient: int, desired patient
    :param substitution_matrix: str, ALL CAOS
    :param num_clusters: int, desired number of clusters in k-means
    :param plot: boolean, toggle UMAP projection plot
    :param kmeans: booelan, toggle k-means clustering plot
    :param louvain: booelan, toggle louvain clustering plot

    """
    sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

    t0 = time.time()
    if all_patients:
        PAT = f'ALL_SEQUENCES_{substitution_matrix}_DISTANCE_MATRIX'
    else:
        PAT = fr'PATIENT_{patient}_{substitution_matrix}_DISTANCE_MATRIX'
    patient_data = joblib.load(rf'/home/ubuntu/Enno/gammaDelta/distance_matrices/{PAT}')

    ix, iy = patient_data.shape
    p1_df = pd.DataFrame(data=patient_data, index=[f'Sequence_{i}' for i in range(1, ix+1)],
                         columns=[f'Sequence_{i}' for i in range(1, iy+1)])

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(p1_df)        # type(embedding) = <class 'numpy.ndarray'

    if plot:
        plt.scatter(embedding[:, 0], embedding[:, 1])
        plt.title(f'UMAP projection of Patient {patient} using {substitution_matrix}')
        plt.show()
        plt.close()

    if kmeans:
        # UMAP with kmeans
        kmeans_labels = cluster.KMeans(n_clusters=num_clusters).fit_predict(p1_df)
        plt.scatter(embedding[:, 0], embedding[:, 1], c=kmeans_labels, cmap='Spectral')

        if all_patients:
            plt.title(fr'Louvain com. det. in UMAP projection of all patients using {substitution_matrix}')
            plt.savefig(fr'/home/ubuntu/Enno/gammaDelta/plots/ALL_PATIENT_{substitution_matrix}_KMEANS.png')
        else:
            plt.title(fr'UMAP projection of Patient {patient} using {substitution_matrix} and k_means with {num_clusters}')
            plt.savefig(fr'/home/ubuntu/Enno/gammaDelta/plots/PATIENT_{patient}_{substitution_matrix}_KMEANS.png')

        plt.show()

    if louvain:

        t0 = time.time()
        # CREATE GRAPH
        G = numpy_to_graph(patient_data)
        t1 = time.time()-t0
        print('graph check')
        print(f'This took {t1} seconds')

        t0 = time.time()
        # LOUVAIN ALGO
        # NETWORKIT  communities = networkit.community.detectCommunities(G, algo=networkit.community.PLM(G, True))
        # NETWORKIT  partition = communities.getSubsetIds()
        partition = community_louvain.best_partition(G)
        print(partition)
        print(partition.values())

        t1 = time.time() - t0
        print("partition check")
        print(f'This took {t1} seconds')

        t0 = time.time()
        # PLOT LOVUAIN CLUSTERING
        cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
        plt.scatter(embedding[:, 0], embedding[:, 1], cmap=cmap, c=list(partition.values()), s=5)


        t1 = time.time() - t0
        print("scatter check")
        print(f'This took {t1} seconds')

        if all_patients:
            plt.title(f'Louvain com. det. in UMAP projection of all patients using {substitution_matrix}')
            plt.savefig(fr'/home/ubuntu/Enno/gammaDelta/plots/ALL_PATIENT_{substitution_matrix}.png')
        else:
            plt.title(f'Louvain com. det. in UMAP projection of patient {patient} using {substitution_matrix}')
            # plt.savefig(fr'/home/ubuntu/Enno/gammaDelta/plots/PATIENT_{patient}_{substitution_matrix}.png')
        # plt.show()


def numpy_to_graph_naive(dist_mat):
    m, _ = dist_mat.shape
    G = networkit.Graph(m, weighted=True)

    mask_x, mask_y = np.mask_indices(m, np.tril, -1)
    masking_zip = zip(mask_x, mask_y, dist_mat[mask_x, mask_y])

    for nodeA, nodeB, weight in masking_zip:
        G.addEdge(nodeA, nodeB, weight)

    return G


def numpy_to_graph(dist_mat):
    g = nx.Graph(weighted=True)

    for ix, row in enumerate(dist_mat):
        for iy, i in enumerate(row):
            g.add_edge(ix, iy, weight=i)

    print(g.number_of_edges())
    print(g.number_of_nodes())

    return g


if __name__ == '__main__':
    print('Lets go')
    plot_umap(1, 'BLOSUM80', all_patients=True, louvain=True)
    print('Lets go again')
    plot_umap(1, 'GONNET1992', all_patients=True, louvain=True)

