from pipeline2 import *

kind = 'absolute'
filename = 'outputfile'


full_df = get_fasta_info()
full_A = get_am(pam70, full=True)
full_A = shift_similarities_to_zero(full_A)  # shifted affinity matrix A

full_A, full_df = exclude_class('HD', full_df, full_A)

patient_ids = [x[0] for x in full_df.index]
indexes = np.unique(patient_ids, return_index=True)[1]
patient_ids = [patient_ids[index] for index in sorted(indexes)]

cv_splits_arr, train_I_arr, test_I_arr = get_matrix_train_test(df=full_df, mat=full_A, n_splits=3, test_size=0.33, alle=False)

print('ready to go')

total_performances = []
N_C = np.geomspace(5, 500, 10).astype(int)  # TODO 20

for ix, n_c in enumerate(N_C):
    try:
        total_split_performance = []
        for split, train_I, test_I in zip(cv_splits_arr, train_I_arr, test_I_arr):
            train_df, train_A, train_Y, test_df, test_A, test_Y = split

            train_C, N_C = get_cluster(graph=None, gamma=1, n_cluster=n_c, affinity_mat=train_A, kind='spectral')
            train_F, train_SPC = get_train_F(train_C, train_df, kind=kind)

            # Here stability selection is instantiated and run
            selector = StabilitySelection(
                base_estimator=LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=50000,
                                                  penalty='l1'), lambda_name='C',
                lambda_grid=np.logspace(-5, -1, 50), bootstrap_func='stratified', n_jobs=-1, threshold=0.3).fit(
                train_F, train_Y)

            selected_ix = selector.get_support(indices=True)
            n_selected_ix = len(selected_ix)

            selected_SPC = np.array(train_SPC, dtype=object)[selected_ix]

            selected_train_F = np.array(train_F)[:, selected_ix]

            print('ss passed')

            # Model Evaluation
            model = LogisticRegression(class_weight='balanced', max_iter=50000)

            train_D, test_D = similarities_to_distances(train_A), similarities_to_distances(test_A)

            model.fit(selected_train_F, train_Y)

            test_C = get_test_C(train_D, train_C, test_D)
            test_F, test_SPC = get_test_F(test_C, test_df, n_cluster=n_c, kind=kind)
            selected_test_F = test_F[:, selected_ix]

            y_pred = model.predict(selected_test_F)
            balacc = balanced_accuracy_score(test_Y, y_pred)
            total_split_performance.append(balacc)

        perf = np.average(total_split_performance)
        total_split_performance.append(perf)
        s = stats.sem(total_split_performance)
        total_split_performance.append(s)
        total_split_performance.append(n_selected_ix)
        print(total_split_performance)
        total_performances.append(total_split_performance)

    except Exception:
        pass

result_df = pd.DataFrame(total_performances, columns=['F1', 'F2', 'F3', 'T', 'SEM', 'N'])
result_df.to_csv(filename)
