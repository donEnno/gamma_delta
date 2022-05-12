from pipeline2 import *

# Globals
path_to_sm = '/home/ubuntu/Enno/mnt/volume/dm_in_use/BLFUHD_PAM70_1_0.1_DM'
substitution_matrix_name = 'PAM70'

N_SCLUSTER = np.geomspace(5, 500, 3).astype(int)  # TODO 20?
GAMMAS = [1.01, 1.08, 1.13]

# Output
performance = []

# Initialize data
full_df = get_fasta_info()
full_A = get_am(path_to_sm, full=True)
full_A = shift_similarities_to_zero(full_A)
full_D = similarities_to_distances(full_A)

for n_s_cluster, gamma in zip(N_SCLUSTER, GAMMAS):
    cv_splits_arr, train_I_arr, test_I_arr = get_matrix_train_test(df=full_df, mat=full_A, n_splits=5, test_size=0.2)

    counter = 1
    for split, train_I, test_I in zip(cv_splits_arr, train_I_arr, test_I_arr):
        print(f'Fold number: {counter}')
        counter += 1

        train_df, train_A, train_Y, test_df, test_A, test_Y = split

        train_D = similarities_to_distances(train_A)
        test_D = similarities_to_distances(test_A)
        train_G = get_graph(train_A)

        # spectral and leiden cluster vectors
        train_spectral_C, n_train_spectral = get_cluster(graph=None, gamma=1, n_cluster=n_s_cluster, affinity_mat=train_A,
                                                         kind='spectral')
        train_leiden_C, n_train_leiden = get_cluster(graph=train_G, gamma=gamma, n_cluster=0, affinity_mat=np.array([]),
                                                     kind='leiden')

        # spectral and leiden relative
        train_spectral_F_rel, _ = get_train_F(train_spectral_C, train_df, kind='relative')
        train_leiden_F_rel, _ = get_train_F(train_leiden_C, train_df, kind='relative')

        test_spectral_C_rel = get_test_C(train_D, train_spectral_C, test_D)
        test_leiden_C_rel = get_test_C(train_D, train_leiden_C, test_D)

        test_spectral_F_rel, _ = get_test_F(test_spectral_C_rel, test_df, n_train_spectral, kind='relative')
        test_leiden_F_rel, _ = get_test_F(test_leiden_C_rel, test_df, n_train_leiden, kind='relative')

        # spectral and leiden absolute
        train_spectral_F_abs, _ = get_train_F(train_spectral_C, train_df, kind='absolute')
        train_leiden_F_abs, _ = get_train_F(train_leiden_C, train_df, kind='absolute')

        test_spectral_C_abs = get_test_C_precomputed(train_D, train_spectral_C, test_D)
        test_leiden_C_abs = get_test_C_precomputed(train_D, train_leiden_C, test_D)

        test_spectral_F_abs, _ = get_test_F(test_spectral_C_abs, test_df, n_train_spectral, kind='absolute')
        test_leiden_F_abs, _ = get_test_F(test_leiden_C_abs, test_df, n_train_leiden, kind='absolute')

