from pipeline2 import *

import pandas as pd
import numpy as np


# Globals
path_to_sm = '/home/ubuntu/Enno/mnt/volume/dm_in_use/BLFUHD_PAM70_1_0.1_DM'
substitution_matrix_name = 'PAM70'

# Cluster parameter
N_SCLUSTER = np.geomspace(5, 500, 3).astype(int)  # TODO 20?
GAMMAS = [1.01, 1.08, 1.13]

"""    
[1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.2]
"""

# Output
performance = []

# Initialize data
full_df = get_fasta_info()
full_A = get_am(path_to_sm, full=True)
full_A = shift_similarities_to_zero(full_A)
full_D = similarities_to_distances(full_A)

for n_s_cluster, gamma in zip(N_SCLUSTER, GAMMAS):
    print('Currently working on {} spectral clusters and gamma {}'.format(n_s_cluster, gamma))
    # 5-fold CV
    cv_splits_arr, train_I_arr, test_I_arr = get_matrix_train_test(df=full_df, mat=full_A, n_splits=5, test_size=0.2)
    leiden_abs_folds, leiden_rel_folds, spectral_abs_folds, spectral_rel_folds = [], [], [], []

    counter = 1
    for split, train_I, test_I in zip(cv_splits_arr, train_I_arr, test_I_arr):
        print('Fold number: {}'.format(counter))
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

        # classification
        model = LogisticRegression(class_weight='balanced', max_iter=50000)

        # spectral absolute
        model.fit(train_spectral_F_abs, train_Y)
        spectral_pred_Y_abs_self = model.predict(train_spectral_F_abs)
        spectral_pred_Y_abs = model.predict(test_spectral_F_abs)

        spectral_abs_train_performance = balanced_accuracy_score(spectral_pred_Y_abs_self, train_Y)
        spectral_abs_test_performance = balanced_accuracy_score(spectral_pred_Y_abs, test_Y)

        spectral_abs_folds.append([spectral_abs_train_performance, spectral_abs_test_performance])

        # spectral relative
        model.fit(train_spectral_F_rel, train_Y)
        spectral_pred_Y_rel_self = model.predict(train_spectral_F_rel)
        spectral_pred_Y_rel = model.predict(test_spectral_F_rel)

        spectral_rel_train_performance = balanced_accuracy_score(spectral_pred_Y_rel_self, train_Y)
        spectral_rel_test_performance = balanced_accuracy_score(spectral_pred_Y_rel, test_Y)

        spectral_rel_folds.append([spectral_rel_train_performance, spectral_rel_test_performance])

        # leiden absolute
        model.fit(train_leiden_F_abs, train_Y)
        leiden_pred_Y_abs_self = model.predict(train_leiden_F_abs)
        leiden_pred_Y_abs = model.predict(test_leiden_F_abs)

        leiden_abs_train_performance = balanced_accuracy_score(leiden_pred_Y_abs_self, train_Y)
        leiden_abs_test_performance = balanced_accuracy_score(leiden_pred_Y_abs, test_Y)

        leiden_abs_folds.append([leiden_abs_train_performance, leiden_abs_test_performance])

        # leiden relative
        model.fit(train_leiden_F_rel, train_Y)
        leiden_pred_Y_rel_self = model.predict(train_leiden_F_rel)
        leiden_pred_Y_rel = model.predict(test_leiden_F_rel)

        leiden_rel_train_performance = balanced_accuracy_score(leiden_pred_Y_rel_self, train_Y)
        leiden_rel_test_performance = balanced_accuracy_score(leiden_pred_Y_rel, test_Y)

        leiden_rel_folds.append([leiden_rel_train_performance, leiden_rel_test_performance])

        print(leiden_rel_folds)
        print(leiden_abs_folds)
        print(spectral_rel_folds)
        print(spectral_abs_folds)

    # leiden rel train
    leiden_rel_entry_train = ['TRAIN', 'LEIDEN', 'REL']
    leiden_rel_entry_train.extend([x[0] for x in leiden_rel_folds])
    leiden_rel_entry_train.extend([np.average([x[0] for x in leiden_rel_folds])])
    leiden_rel_entry_train.extend([n_train_leiden])
    # leiden rel test
    leiden_rel_entry_test = ['TEST', 'LEIDEN', 'REL']
    leiden_rel_entry_test.extend([x[1] for x in leiden_rel_folds])
    leiden_rel_entry_test.extend([np.average([x[1] for x in leiden_rel_folds])])
    leiden_rel_entry_test.extend([n_train_leiden])
    # leiden abs train
    leiden_abs_entry_train = ['TRAIN', 'LEIDEN', 'ABS']
    leiden_abs_entry_train.extend([x[0] for x in leiden_abs_folds])
    leiden_abs_entry_train.extend([np.average([x[0] for x in leiden_abs_folds])])
    leiden_abs_entry_train.extend([n_train_leiden])
    # leiden abs test
    leiden_abs_entry_test = ['TEST', 'LEIDEN', 'ABS']
    leiden_abs_entry_test.extend([x[1] for x in leiden_abs_folds])
    leiden_abs_entry_test.extend([np.average([x[1] for x in leiden_abs_folds])])
    leiden_abs_entry_test.extend([n_train_leiden])
    # spectral rel train
    spectral_rel_entry_train = ['TRAIN', 'SPECTRAL', 'REL']
    spectral_rel_entry_train.extend([x[0] for x in spectral_rel_folds])
    spectral_rel_entry_train.extend([np.average([x[0] for x in spectral_rel_folds])])
    spectral_rel_entry_train.extend([n_train_spectral])
    # spectral rel test
    spectral_rel_entry_test = ['TEST', 'SPECTRAL', 'REL']
    spectral_rel_entry_test.extend([x[1] for x in spectral_rel_folds])
    spectral_rel_entry_test.extend([np.average([x[1] for x in spectral_rel_folds])])
    spectral_rel_entry_test.extend([n_train_spectral])
    # spectral abs train
    spectral_abs_entry_train = ['TRAIN', 'SPECTRAL', 'ABS']
    spectral_abs_entry_train.extend([x[0] for x in spectral_abs_folds])
    spectral_abs_entry_train.extend([np.average([x[0] for x in spectral_abs_folds])])
    spectral_abs_entry_train.extend([n_train_spectral])
    # spectral abs test
    spectral_abs_entry_test = ['TEST', 'SPECTRAL', 'ABS']
    spectral_abs_entry_test.extend([x[1] for x in spectral_abs_folds])
    spectral_abs_entry_test.extend([np.average([x[1] for x in spectral_abs_folds])])
    spectral_abs_entry_test.extend([n_train_spectral])

    performance.append(leiden_rel_entry_train)
    performance.append(leiden_rel_entry_test)
    performance.append(leiden_abs_entry_train)
    performance.append(leiden_abs_entry_test)
    performance.append(spectral_rel_entry_train)
    performance.append(spectral_rel_entry_test)
    performance.append(spectral_abs_entry_train)
    performance.append(spectral_abs_entry_test)

# results to file
performance_df = pd.DataFrame(performance, columns=['P_TYPE', 'CLUSTER_T', 'FEATURE_T', 'F1', 'F2', 'F3', 'F4', 'F5', 'AVG_F', 'N'])
performance_df.to_csv('leiden_spectral_performance_revisited.csv')
