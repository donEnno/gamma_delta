from pipeline2 import *

# Globals
path_to_sm = '/home/ubuntu/Enno/mnt/volume/dm_in_use/BLFUHD_PAM70_1_0.1_DM'
substitution_matrix_name = 'PAM70'

# Initialize data
full_df = get_fasta_info()
full_A = get_am(path_to_sm, full=True)
full_A = shift_similarities_to_zero(full_A)
full_D = similarities_to_distances(full_A)

# Initialize test data
test_P = ['BL-1-', 'BL-2-', 'BL-3-', 'FU-1-', 'FU-2-', 'HD-2-']
testing_df = full_df[full_df.P.isin(test_P)]


# Number of clusters parameters
N_SCLUSTER = np.geomspace(5, 500, 3).astype(int)
GAMMAS = [1.01, 1.08, 1.13]

for n_s_cluster, gamma in zip(N_SCLUSTER, GAMMAS):
    # Set ground truths
    full_G = get_graph(full_A)
    full_leiden_C, n_leiden_C = get_cluster(graph=full_G, gamma=gamma, n_cluster=0, affinity_mat=np.array([]),
                                            kind='leiden')

    full_spectral_C, n_spectral_C = get_cluster(graph=None, gamma=1, n_cluster=n_s_cluster, affinity_mat=full_A,
                                                kind='spectral')

    cv_splits_arr, train_I_arr, test_I_arr = get_matrix_train_test(df=full_df, mat=full_A, n_splits=5, test_size=0.2)

    # Iterate through folds
    counter = 1
    for split, train_I, test_I in zip(cv_splits_arr, train_I_arr, test_I_arr):
        print(f'Fold number: {counter}')
        counter += 1

        train_df, train_A, train_Y, test_df, test_A, test_Y = split

        # Init train data
        train_D = similarities_to_distances(train_A)
        test_D = similarities_to_distances(test_A)
        train_G = get_graph(train_A)

        # Train spectral and leiden cluster vectors
        train_spectral_C, n_train_spectral = get_cluster(graph=None, gamma=1, n_cluster=n_s_cluster, affinity_mat=train_A,
                                                         kind='spectral')
        train_leiden_C, n_train_leiden = get_cluster(graph=train_G, gamma=gamma, n_cluster=0, affinity_mat=np.array([]),
                                                     kind='leiden')

        # Train cluster
        test_spectral_C = get_test_C(train_D, train_spectral_C, test_D)
        test_leiden_C = get_test_C(train_D, train_leiden_C, test_D)

        test_spectral_C_pc = get_test_C_precomputed(train_D, train_spectral_C, test_D)
        test_leiden_C_pc = get_test_C_precomputed(train_D, train_leiden_C, test_D)

        # Combine
        joined_leiden = []
        joined_spectral = []
        joined_leiden_pc = []
        joined_spectral_pc = []

        # Test spectral and leiden cluster vectors
        leiden_ari = adjusted_rand_score(full_leiden_C, joined_leiden)
        spectral_ari = adjusted_rand_score(full_spectral_C, joined_spectral)

        # Store values for every fold and write to dataframe for later translation to csv

        # Save to csv


