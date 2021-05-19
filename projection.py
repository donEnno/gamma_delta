# Default
import os.path
import time
from joblib import Parallel, delayed, dump, load


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
import community as community_louvain
import matplotlib.cm as cm
import networkx as netx

import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
# import hdbscan


def plot_umap(patient, substitution_matrix, num_clusters=5, plot=False, kmeans=False, louvain=False):
    """
    :param patient: int, desired patient
    :param substitution_matrix: str, ALL CAOS
    :param num_clusters: int, desired number of clusters in k-means
    :param plot: boolean, toggle UMAP projection plot
    :param kmeans: booelan, toggle k-means clustering plot
    :param louvain: booelan, toggle louvain clustering plot

    :raises ValueError for patient=2, substitution_matrix='BLOSUM45','GONNET1992'

    """
    sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

    t0 = time.time()
    patient1_data = load(rf'/home/ubuntu/Enno/gammaDelta/distance_matrices/PATIENT_{patient}_{substitution_matrix}_DISTANCE_MATRIX')
    nx, ny = patient1_data.shape
    p1_df = pd.DataFrame(data=patient1_data, index=[f'Sequence_{i}' for i in range(1, nx+1)],
                                           columns=[f'Sequence_{i}' for i in range(1, ny+1)])

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
        plt.title(f'UMAP projection of Patient {patient} using {substitution_matrix} and k_means with {num_clusters}')
        plt.show()

    if louvain:
        G = netx.from_numpy_array(patient1_data)

        # degrees = G.degree()
        # print(dict(degrees).values
        # degrees = G.degree(weight='weight')
        # print(dict(degrees).values())

        partition = community_louvain.best_partition(G)

        # plot the graph
        cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
        plt.scatter(embedding[:, 0], embedding[:, 1], c=list(partition.values()), cmap=cmap)
        plt.title(f'Louvain com. det. in UMAP projection of Patient {patient} using {substitution_matrix}')
        plt.show()


if __name__ == '__main__':
    pass
