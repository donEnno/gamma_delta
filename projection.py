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


def plot_umap(patient, substitution_matrix, num_clusters=5, plot=False, kmeans=False, louvain=False):
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
    # ALL_SEQUENCES_BLOSUM45_DISTANCE_MATRIX
    patient_data = load(rf'/home/ubuntu/Enno/gammaDelta/distance_matrices/PATIENT_{patient}_{substitution_matrix}_DISTANCE_MATRIX')
    # patient_data = load(r'C:\Users\Enno\PycharmProjects\gamma_delta\data\distance_matrices\ALL_SEQUENCES_BLOSUM45_DISTANCE_MATRIX')
    nx, ny = patient_data.shape
    p1_df = pd.DataFrame(data=patient_data, index=[f'Sequence_{i}' for i in range(1, nx+1)],
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
        # print("louvain check")
        G = netx.from_numpy_array(patient_data)
        # print("G check")

        # degrees = G.degree()
        # print(dict(degrees).values)
        # degrees = G.degree(weight='weight')
        # print(dict(degrees).values())

        partition = community_louvain.best_partition(G)
        # print("partition check")
        # plot the graph
        cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
        plt.scatter(embedding[:, 0], embedding[:, 1], c=list(partition.values()), cmap=cmap)
        # print("scatter check")
        plt.title(f'Louvain com. det. in UMAP projection of Patient {patient} using {substitution_matrix}')
        plt.savefig(fr'/home/ubuntu/Enno/gammaDelta/plots/PATIENT_{patient}_{substitution_matrix}.png')
        plt.show()


def numpy_to_graph(ix, row):
    # TODO Method that adds each row to the graph to then parallelize it.
    g = netx.Graph()

    for iy, i in enumerate(row):
        g.add_edge(ix, iy, Weight=i)

    return g


if __name__ == '__main__':

    for patient in range(1, 30):
        for matrix in ['BLOSUM45', 'BLOSUM80', 'GONNET1992']:
            try:
                plot_umap(patient, matrix, louvain=True)
            except ValueError:
                pass

    """
    LOCATE ValueError
    patient2_B45_data = load(rf'/home/ubuntu/Enno/gammaDelta/distance_matrices/PATIENT_2_BLOSUM45_DISTANCE_MATRIX')
    patient2_B80_data = load(rf'/home/ubuntu/Enno/gammaDelta/distance_matrices/PATIENT_2_BLOSUM80_DISTANCE_MATRIX')
    patient2_G92_data = load(rf'/home/ubuntu/Enno/gammaDelta/distance_matrices/PATIENT_2_GONNET1992_DISTANCE_MATRIX')

    G_P2_B45 = netx.from_numpy_array(patient2_B45_data)
    G_P2_B80 = netx.from_numpy_array(patient2_B80_data)
    G_P2_G92 = netx.from_numpy_array(patient2_G92_data)

    degrees_B45 = G_P2_B45.degree()
    degrees_B80 = G_P2_B80.degree()
    degrees_G92 = G_P2_G92.degree()

    print(dict((k, v) for k, v in degrees_B45 if v <= 0))
    print(dict((k, v) for k, v in degrees_B80 if v <= 0))
    print(dict((k, v) for k, v in degrees_G92 if v <= 0))

    # result = dict((k, v) for k, v in ini_dict.items() if v >= 0)

    weights_B45 = G_P2_B45.degree(weight='weight')
    weights_B80 = G_P2_B80.degree(weight='weight')
    weights_G92 = G_P2_G92.degree(weight='weight')

    print(dict((k, v) for k, v in weights_B45 if v <= 0))
    print(dict((k, v) for k, v in weights_B80 if v <= 0))
    print(dict((k, v) for k, v in weights_G92 if v <= 0))"""
