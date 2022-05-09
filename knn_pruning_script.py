from pipeline2 import *
substitution_matrix_name, path_to_sm = 'PAM70', pam70

K = [x/10 for x in list(range(3, 10))]             # TODO
N_SCLUSTER = np.geomspace(5, 500, 20).astype(int)  # TODO 20
# [3, 6, 9, 12, 16, 24, 32, 48, 54, 66, 90, 120, 240, 320, 480, 600, 700, 800, 1000, 1250, 1500]
# GAMMAS = [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17,
#           1.18, 1.19, 1.2]
# cluster_parameter_list = list(zip(N_SCLUSTER, GAMMAS))

performance = []
self_performance = []
ari = []

full_df = get_fasta_info()
full_A = get_am(path_to_sm, full=True)
full_A = shift_similarities_to_zero(full_A)  # shifted affinity matrix A
full_D = similarities_to_distances(full_A)

full_A, reduced_df = exclude_class('FU', full_df, full_A)
full_D, _ = exclude_class('FU', full_df, full_D)

# full_G = get_graph(full_A)

for n_s_cluster in N_SCLUSTER:  # , gamma
    full_spectral_C, n_full_spectral = get_cluster(graph=None, gamma=1, n_cluster=n_s_cluster, affinity_mat=full_A,
                                                   kind='spectral')
    # full_leiden_C, n_full_leiden = get_cluster(graph=full_G, gamma=gamma, n_cluster=0, affinity_mat=np.array([]),
    #                                            kind='leiden')

    for k_ in K:
        kNN_A = kNN_selection(full_A, k_, kind='affinity')
        # kNN_D = shift_similarities_to_zero(kNN_A)
        # kNN_G = get_graph(kNN_A)

        kNN_full_spectral_C, n_kNN_full_spectral = get_cluster(graph=None, gamma=1, n_cluster=n_s_cluster,
                                                               affinity_mat=kNN_A, kind='spectral')
        # kNN_full_leiden_C, n_kNN_full_leiden = get_cluster(graph=kNN_G, gamma=gamma, n_cluster=0,
        #                                                    affinity_mat=np.array([]), kind='leiden')

        # FOR SPLIT IN SPLITS
        cv_splits_arr, train_I_arr, test_I_arr = get_matrix_train_test(df=reduced_df, mat=kNN_A, n_splits=5, test_size=0.2)
        total_split_rel_performance, total_split_abs_performance = [], []

        for split, train_I, test_I in zip(cv_splits_arr, train_I_arr, test_I_arr):
            train_df, kNN_train_A, train_Y, test_df, kNN_test_A, test_Y = split

            kNN_train_D = similarities_to_distances(kNN_train_A)
            kNN_test_D = similarities_to_distances(kNN_test_A)
            # kNN_train_G = get_graph(kNN_train_A)

            # spectral and leiden cluster vectors
            kNN_train_spectral_C, n_kNN_train_spectral = get_cluster(graph=None, gamma=1, n_cluster=n_s_cluster,
                                                                     affinity_mat=kNN_train_A, kind='spectral')
            # kNN_train_leiden_C, n_kNN_train_leiden = get_cluster(graph=kNN_train_G, gamma=gamma, n_cluster=0,
            #                                                      affinity_mat=np.array([]), kind='leiden')

            # spectral and leiden relative
            kNN_train_spectral_F_rel, _ = get_train_F(kNN_train_spectral_C, train_df, kind='relative')
            # kNN_train_leiden_F_rel, _ = get_train_F(kNN_train_leiden_C, train_df, kind='relative')

            kNN_test_spectral_C_rel = get_test_C(kNN_train_D, kNN_train_spectral_C, kNN_test_D)
            # kNN_test_leiden_C_rel = get_test_C(kNN_train_D, kNN_train_leiden_C, kNN_test_D)

            kNN_test_spectral_F_rel, _ = get_test_F(kNN_test_spectral_C_rel, test_df, n_kNN_train_spectral,
                                                    kind='relative')
            # kNN_test_leiden_F_rel, _ = get_test_F(kNN_test_leiden_C_rel, test_df, n_kNN_train_leiden, kind='relative')

            # spectral and leiden absolute
            kNN_train_spectral_F_abs, _ = get_train_F(kNN_train_spectral_C, train_df, kind='absolute')
            # kNN_train_leiden_F_abs, _ = get_train_F(kNN_train_leiden_C, train_df, kind='absolute')

            kNN_test_spectral_C_abs = get_test_C(kNN_train_D, kNN_train_spectral_C, kNN_test_D)
            # kNN_test_leiden_C_abs = get_test_C(kNN_train_D, kNN_train_leiden_C, kNN_test_D)

            kNN_test_spectral_F_abs, _ = get_test_F(kNN_test_spectral_C_abs, test_df, n_kNN_train_spectral,
                                                    kind='absolute')
            # kNN_test_leiden_F_abs, _ = get_test_F(kNN_test_leiden_C_abs, test_df, n_kNN_train_leiden, kind='absolute')

            # classification
            model = LogisticRegression(class_weight='balanced', max_iter=50000)

            # spectral relative and absolute
            model.fit(kNN_train_spectral_F_abs, train_Y)
            kNN_spectral_pred_Y_abs_self = model.predict(kNN_train_spectral_F_abs)
            kNN_spectral_pred_Y_abs = model.predict(kNN_test_spectral_F_abs)

            model.fit(kNN_train_spectral_F_rel, train_Y)
            kNN_spectral_pred_Y_rel_self = model.predict(kNN_train_spectral_F_rel)
            kNN_spectral_pred_Y_rel = model.predict(kNN_test_spectral_F_rel)

            total_split_abs_performance.append([balanced_accuracy_score(test_Y, kNN_spectral_pred_Y_abs),
                                               f1_score(test_Y, kNN_spectral_pred_Y_abs),
                                               precision_score(test_Y, kNN_spectral_pred_Y_abs),
                                               recall_score(test_Y, kNN_spectral_pred_Y_abs),
                                               recall_score(test_Y, kNN_spectral_pred_Y_abs, pos_label=0)])

            total_split_rel_performance.append([balanced_accuracy_score(test_Y, kNN_spectral_pred_Y_rel),
                                                f1_score(test_Y, kNN_spectral_pred_Y_rel),
                                                precision_score(test_Y, kNN_spectral_pred_Y_rel),
                                                recall_score(test_Y, kNN_spectral_pred_Y_rel),
                                                recall_score(test_Y, kNN_spectral_pred_Y_rel, pos_label=0)])
            print(total_split_rel_performance)
            print(total_split_abs_performance)

        abs_performance = np.average(total_split_abs_performance, axis=0)
        rel_performance = np.average(total_split_rel_performance, axis=0)

        print(abs_performance)
        print(rel_performance)

        abs_entry = ['test', k_, 'spectral', 'abs', n_s_cluster]
        abs_entry.extend(abs_performance)
        rel_entry = ['test', k_, 'spectral', 'rel', n_s_cluster]
        rel_entry.extend(rel_performance)

        performance.append(abs_entry)
        performance.append(rel_entry)

# results to file
performance_df = pd.DataFrame(performance, columns=['TYPE', 'K', 'CK', 'FK', 'NC', 'BA', 'F1', 'PR', 'SP', 'SN'])
self_performance_df = pd.DataFrame(self_performance, columns=['TYPE', 'K', 'CK', 'FK', 'NC', 'BA', 'F1', 'PR', 'SP', 'SN'])
ari_df = pd.DataFrame(ari, columns=['CK', 'FK', 'K', 'FvJ', 'KvJ'])

performance_df.to_csv('{}/{}_k_run_test_performance.csv'.format(substitution_matrix_name, substitution_matrix_name))
self_performance_df.to_csv('{}/{}_k_run_train_performance.csv'.format(substitution_matrix_name, substitution_matrix_name))
ari_df.to_csv('{}/{}_k_run_ari.csv'.format(substitution_matrix_name, substitution_matrix_name))
