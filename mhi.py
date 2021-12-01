import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def morista_horn(X: list, Y: list, S: list):
    z = 0
    n = 0
    m = 0
    x = len(X)
    y = len(Y)
    for unique_sequence in S:
        z = z + X.count(unique_sequence) * Y.count(unique_sequence)
        n = n + X.count(unique_sequence) ** 2
        m = m + Y.count(unique_sequence) ** 2
    C = 2 * z / (((n / x ** 2) + (m / y ** 2)) * x * y)
    return C


def unique_samples(a, b):
    u_list = a + b
    u_list = list(np.unique(np.array(u_list)))
    u_list = [str(s) for s in u_list]
    return u_list


def replace_inf(intra, lower):
    for ix, x in enumerate(intra):
        if x < -1000:
            intra[ix] = lower


def show_min(intra):
    return min([x for x in intra if x > -10000])


def eliminate_inf(intra):
    return [x for x in intra if x > -7]



path_sick = '/home/ubuntu/Enno/gammaDelta/data/sick'
path_hd = '/home/ubuntu/Enno/gammaDelta/data/healthy'
path_bl = '/home/ubuntu/Enno/gammaDelta/data/BL_txt'
path_fu = '/home/ubuntu/Enno/gammaDelta/data/FU_txt'

s_files = os.listdir(path_sick)
h_files = os.listdir(path_hd)
b_files = os.listdir(path_bl)
f_files = os.listdir(path_fu)

sick_df = pd.read_csv(path_sick + '/' + s_files[0], sep='\t')
for file in s_files[1:]:
    path = path_sick + '/' + file
    temp = pd.read_csv(path, sep='\t')
    sick_df = sick_df.append(temp)

hd_df = pd.read_csv(path_hd + '/' + h_files[0], sep='\t')
for file in h_files[1:]:
    path = path_hd + '/' + file
    temp = pd.read_csv(path, sep='\t')
    hd_df = hd_df.append(temp)

bl_df = pd.read_csv(path_bl + '/' + b_files[0], sep='\t')
for file in b_files[1:]:
    path = path_bl + '/' + file
    temp = pd.read_csv(path, sep='\t')
    bl_df = bl_df.append(temp)

fu_df = pd.read_csv(path_fu + '/' + f_files[0], sep='\t')
for file in f_files[1:]:
    path = path_fu + '/' + file
    temp = pd.read_csv(path, sep='\t')
    fu_df = fu_df.append(temp)

sick_df.reset_index(drop=True, inplace=True)
bl_df.reset_index(drop=True, inplace=True)
hd_df.reset_index(drop=True, inplace=True)
fu_df.reset_index(drop=True, inplace=True)

si_list = list(sick_df.to_records(index=False))
hd_list = list(hd_df.to_records(index=False))
bl_list = list(bl_df.to_records(index=False))
fu_list = list(fu_df.to_records(index=False))

else_hd_list = [s for s, v, c, f in hd_list if (v != 'TRDV2' and v != 'TRDV1' and v != 'TRDV3')]
else_si_list = [s for s, v, c, f in si_list if (v != 'TRDV2' and v != 'TRDV1' and v != 'TRDV3')]
else_bl_list = [s for s, v, c, f in bl_list if (v != 'TRDV2' and v != 'TRDV1' and v != 'TRDV3')]
else_fu_list = [s for s, v, c, f in fu_list if (v != 'TRDV2' and v != 'TRDV1' and v != 'TRDV3')]

v1_hd_list = [s for s, v, c, f in hd_list if v == 'TRDV1']
v1_si_list = [s for s, v, c, f in si_list if v == 'TRDV1']
v1_bl_list = [s for s, v, c, f in bl_list if v == 'TRDV1']
v1_fu_list = [s for s, v, c, f in fu_list if v == 'TRDV1']
v1_blfu_list = v1_bl_list + v1_fu_list

v2_hd_list = [s for s, v, c, f in hd_list if v == 'TRDV2']
v2_si_list = [s for s, v, c, f in si_list if v == 'TRDV2']
v2_bl_list = [s for s, v, c, f in bl_list if v == 'TRDV2']
v2_fu_list = [s for s, v, c, f in fu_list if v == 'TRDV2']
v2_blfu_list = v2_bl_list + v2_fu_list

v3_hd_list = [s for s, v, c, f in hd_list if v == 'TRDV3']
v3_si_list = [s for s, v, c, f in si_list if v == 'TRDV3']
v3_bl_list = [s for s, v, c, f in bl_list if v == 'TRDV3']
v3_fu_list = [s for s, v, c, f in fu_list if v == 'TRDV3']
v3_blfu_list = v3_bl_list + v3_fu_list

subsets = [v1_hd_list, v1_si_list, v1_bl_list, v1_fu_list, v2_hd_list, v2_si_list, v2_bl_list, v2_fu_list, v3_hd_list,
           v3_si_list, v3_bl_list, v3_fu_list, else_hd_list, else_si_list, else_bl_list, else_fu_list]
labels = ['v1_hd', 'v1_si', 'v1_bl', 'v1_fu', 'v2_hd', 'v2_si', 'v2_bl', 'v2_fu', 'v3_hd', 'v3_si', 'v3_bl', 'v3_fu',
          'else_hd_list', 'else_si_list', 'else_bl_list', 'else_fu_list']

samples = list(zip(subsets, labels))

bl_pats = []
for file in b_files:
    path = path_bl + '/' + file
    bl_pats.append(pd.read_csv(path, sep='\t'))

fu_pats = []
for file in f_files:
    path = path_fu + '/' + file
    fu_pats.append(pd.read_csv(path, sep='\t'))

hd_pats = []
for file in h_files:
    path = path_hd + '/' + file
    hd_pats.append(pd.read_csv(path, sep='\t'))

blfu_pats = bl_pats + fu_pats

hd_list = [s for s, v, c, f in hd_list]
bl_list = [s for s, v, c, f in bl_list]
fu_list = [s for s, v, c, f in fu_list]
bl
fu_list = bl_list + fu_list

groups = [hd_list, bl_list, fu_list, blfu_list]

hd_pan = []
ct = 0
for p_i in hd_pats:
    ct += 1
    print(ct)
    p_i = p_i['cdr3aa'].to_list()
    s = unique_samples(p_i, hd_list)
    c = morista_horn(p_i, hd_list, s)
    hd_pan.append(c)

bl_pan = []
ct = 0
for p_i in bl_pats:    ct += 1
    print(ct)
    p_i = p_i['cdr3aa'].to_list()
    s = unique_samples(p_i, bl_list)
    c = morista_horn(p_i, bl_list, s)
    bl_pan.append(c)

fu_pan = []
ct = 0
for p_i in fu_pats:
    ct += 1
    print(ct)
    p_i = p_i['cdr3aa'].to_list()
    s = unique_samples(p_i, fu_list)
    c = morista_horn(p_i, fu_list, s)
    fu_pan.append(c)

blfu_pan = []
ct = 0
for p_i in blfu_pats:
    ct += 1
    print(ct)
    p_i = p_i['cdr3aa'].to_list()
    s = unique_samples(p_i, blfu_list)
    c = morista_horn(p_i, blfu_list, s)
    blfu_pan.append(c)

hd_v1 = []
ct = 0
for p_i in hd_pats:
    ct += 1
    print(ct)
    p_i = p_i['cdr3aa'].to_list()
    s = unique_samples(p_i, v1_hd_list)
    c = morista_horn(p_i, v1_hd_list, s)
    hd_v1.append(c)

hd_v2 = []
ct = 0
for p_i in hd_pats:
    ct += 1
    print(ct)
    p_i = p_i['cdr3aa'].to_list()
    s = unique_samples(p_i, v2_hd_list)
    c = morista_horn(p_i, v2_hd_list, s)
    hd_v2.append(c)

hd_v3 = []
ct = 0
for p_i in hd_pats:
    ct += 1
    print(ct)
    p_i = p_i['cdr3aa'].to_list()
    s = unique_samples(p_i, v3_hd_list)
    c = morista_horn(p_i, v3_hd_list, s)
    hd_v3.append(c)

bl_v1 = []
ct = 0
for p_i in bl_pats:
    ct += 1
    print(ct)
    p_i = p_i['cdr3aa'].to_list()
    s = unique_samples(p_i, v1_bl_list)
    c = morista_horn(p_i, v1_bl_list, s)
    bl_v1.append(c)

bl_v2 = []
ct = 0
for p_i in bl_pats:
    ct += 1
    print(ct)
    p_i = p_i['cdr3aa'].to_list()
    s = unique_samples(p_i, v2_bl_list)
    c = morista_horn(p_i, v2_bl_list, s)
    bl_v2.append(c)

bl_v3 = []
ct = 0
for p_i in bl_pats:
    ct += 1
    print(ct)
    p_i = p_i['cdr3aa'].to_list()
    s = unique_samples(p_i, v3_bl_list)
    c = morista_horn(p_i, v3_bl_list, s)
    bl_v3.append(c)

fu_v1 = []
ct = 0
for p_i in fu_pats:
    ct += 1
    print(ct)
    p_i = p_i['cdr3aa'].to_list()
    s = unique_samples(p_i, v1_fu_list)
    c = morista_horn(p_i, v1_fu_list, s)
    fu_v1.append(c)

fu_v2 = []
ct = 0
for p_i in fu_pats:
    ct += 1
    print(ct)
    p_i = p_i['cdr3aa'].to_list()
    s = unique_samples(p_i, v2_fu_list)
    c = morista_horn(p_i, v2_fu_list, s)
    fu_v2.append(c)

fu_v3 = []
ct = 0
for p_i in fu_pats:
    ct += 1
    print(ct)
    p_i = p_i['cdr3aa'].to_list()
    s = unique_samples(p_i, v3_fu_list)
    c = morista_horn(p_i, v3_fu_list, s)
    fu_v3.append(c)

blfu_v1 = []
ct = 0
for p_i in blfu_pats:
    ct += 1
    print(ct)
    p_i = p_i['cdr3aa'].to_list()
    s = unique_samples(p_i, v1_blfu_list)
    c = morista_horn(p_i, v1_blfu_list, s)
    blfu_v1.append(c)

blfu_v2 = []
ct = 0
for p_i in blfu_pats:
    ct += 1
    print(ct)
    p_i = p_i['cdr3aa'].to_list()
    s = unique_samples(p_i, v2_blfu_list)
    c = morista_horn(p_i, v2_blfu_list, s)
    blfu_v2.append(c)

blfu_v3 = []
ct = 0
for p_i in blfu_pats:
    ct += 1
    print(ct)
    p_i = p_i['cdr3aa'].to_list()
    s = unique_samples(p_i, v3_blfu_list)
    c = morista_horn(p_i, v3_blfu_list, s)
    blfu_v3.append(c)

pan_df = hd_pan + bl_pan + fu_pan + blfu_pan
v1_df = hd_v1 + bl_v1 + fu_v1 + blfu_v1
v2_df = hd_v2 + bl_v2 + fu_v2 + blfu_v2
v3_df = hd_v3 + bl_v3 + fu_v3 + blfu_v3
labels = ['HD'] * 29 + ['BL'] * 66 + ['FU'] * 55 + ['BLFU'] * 121
pan_df = pd.DataFrame({'MHI': pan_df, 'label': labels})
v1_df = pd.DataFrame({'MHI': v1_df, 'label': labels})
v2_df = pd.DataFrame({'MHI': v2_df, 'label': labels})
v3_df = pd.DataFrame({'MHI': v3_df, 'label': labels})




labels = ['HD']*len(results[0]) + ['BL']*len(results[1]) + ['FU']*len(results[2]) + ['BLFU']*len(results[3]) + ['BLFU pairs']*len(results[4]) + ['BLFU no pairs']*len(results[5])
flat_results = [x for y in results for x in y]
results_df = pd.DataFrame({'MHI': flat_results, 'label': labels})
groups_only = results[:4]
blfu_only = result[3:]
blfu_only = results[3:]
groups_only_labels = ['HD']*len(results[0]) + ['BL']*len(results[1]) + ['FU']*len(results[2]) + ['BLFU']*len(results[3])
blfu_only_labels = ['BLFU']*len(results[3]) + ['BLFU pairs']*len(results[4]) + ['BLFU no pairs']*len(results[5])
groups_only_flat = [x for y in groups_only for x in y]
blfu_only_flat = [x for y in blfu_only for x in y]
groups_only_df = pd.DataFrame({'MHI': groups_only_flat, 'label': groups_only_labels})
blfu_only_df = pd.DataFrame({'MHI': blfu_only_flat, 'label': blfu_only_labels})

list_of_results_folder = os.listdir(root)
print(list_of_results_folder.sort())
for ix, model in enumerate(list_of_results_folder):
    neg_seq = []
    pos_seq = []
    print(model)
    model_root = root + model
    list_of_results = list(os.listdir(model_root))
    for file in list_of_results:
        if 'fasta' in file:
            path = model_root + '/' + file
            with open(path, 'r') as feature:
                lines = feature.readlines()
                if 'neg' in file:
                    for line in lines:
                        if line.startswith('>'):
                            neg_seq.append(line)
                if 'pos' in file:
                    for line in lines:
                        if line.startswith('>'):
                            pos_seq.append(line)
    common_seq = list(set(pos_seq).intersection(set(neg_seq)))
    print('#NEG: ', len(neg_seq))
    print('#POS: ', len(pos_seq))
    print('#COMMON:', len(common_elements))
    print('#FRACTION: %.3f' % (len(common_elements) / max(len(neg_seq), len(pos_seq))))
    print()
    df_row = [model, len(neg_seq), len(pos_seq), len(common_seq), len(common_seq) / max(len(neg_seq), len(pos_seq))]
    results.loc[ix] = df_row


def global_overlap():
    overlap_df = pd.DataFrame(columns=['model', 'number_neg_seq', 'number_pos_seq', 'number_common_seq', 'ratio'])
    root = "/home/ubuntu/Enno/mnt/volume/results/"

    list_of_results_folder = os.listdir(root)
    # print(list_of_results_folder.sort())

    for ix, model in enumerate(list_of_results_folder):
        neg_seq = []
        pos_seq = []
        # print(model)
        model_root = root + model
        list_of_results = list(os.listdir(model_root))
        for file in list_of_results:
            if 'fasta' in file:
                path = model_root + '/' + file
                with open(path, 'r') as feature:
                    lines = feature.readlines()
                    if 'neg' in file:
                        for line in lines:
                            if line.startswith('>'):
                                neg_seq.append(line)
                    if 'pos' in file:
                        for line in lines:
                            if line.startswith('>'):
                                pos_seq.append(line)
        common_seq = list(set(pos_seq).intersection(set(neg_seq)))
        # print('#NEG: ', len(neg_seq))
        # print('#POS: ', len(pos_seq))
        # print('#COMMON:', len(common_seq))
        # print('#FRACTION: %.3f' % (len(common_seq) / max(len(neg_seq), len(pos_seq))))
        # print()
        df_row = [model, len(neg_seq), len(pos_seq), len(common_seq), len(common_seq) / max(len(neg_seq), len(pos_seq))]
        overlap_df.loc[ix] = df_row
    print(overlap_df)


def pairwise_overlap():
    list_of_overlap_df = []
    root = "/home/ubuntu/Enno/mnt/volume/results/"

    list_of_results_folder = os.listdir(root)
    list_of_results_folder.sort()

    for ix, model in enumerate(list_of_results_folder):  # MODEL
        neg_seq = []
        pos_seq = []
        # print(model)
        model_root = root + model
        list_of_results = list(os.listdir(model_root))
        list_of_results = [res for res in list_of_results if 'fasta' in res]
        list_of_results.sort()
        overlap_mat = np.zeros((len(list_of_results), len(list_of_results)))
        columns = []
        neg_c = 0
        pos_c = 0
        # print(list_of_results)
        for i, file in enumerate(list_of_results):
            # print(file) # CLUSTER_1
            if 'neg' in file:
                neg_c += 1
                columns.append(f'neg_{neg_c}')
            else:
                pos_c += 1
                columns.append(f'pos_{pos_c}')
            c1_seq = []
            path = model_root + '/' + file
            with open(path, 'r') as feature_1:
                lines = feature_1.readlines()
                for line in lines:
                    if line.startswith('>'):
                        c1_seq.append(line)

            for j, other_file in enumerate(list_of_results):
                c2_seq = []
                other_path = model_root + '/' + other_file
                with open(other_path, 'r') as feature_2:
                    lines = feature_2.readlines()
                    for line in lines:
                        if line.startswith('>'):
                            c2_seq.append(line)
                cluster_overlap = len(list(set(c1_seq).intersection(set(c2_seq)))) / min(len(c1_seq), len(c2_seq))
                overlap_mat[i, j] = round(cluster_overlap, 3)

        overlap_df = pd.DataFrame(overlap_mat, columns=columns, index=columns)
        list_of_overlap_df.append((model, overlap_df))
    for name, df in list_of_overlap_df:
        print(name + '\\\\')
        print(df.to_latex())
        print('\\\\')


def compute_h_d_ratio():
    root = "/home/ubuntu/Enno/mnt/volume/results/"

    list_of_results_folder = os.listdir(root)
    list_of_results_folder.sort()
    list_of_df = []

    for ix, model in enumerate(list_of_results_folder):  # MODELS

        # print(model)
        model_root = root + model
        list_of_results = list(os.listdir(model_root))
        list_of_results.sort()
        list_of_results = [res for res in list_of_results if 'fasta' in res]
        neg_temp = []
        pos_temp = []

        num_neg = 0
        num_pos = 0

        ratio_per_cluster_df = pd.DataFrame(columns=['SIGN', '#HEALTHY', '#DISEASED', 'RATIO', 'FRAC #H'])

        for jx, file in enumerate(list_of_results):  # CLUSTER
            if 'fasta' in file:
                path = model_root + '/' + file
                with open(path, 'r') as feature:
                    lines = feature.readlines()

                    if 'neg' in file:  # NEG
                        num_neg += 1
                        neg_h = 0
                        neg_s = 0
                        for line in lines:
                            if line.startswith('>'):
                                if 'HD' in line:
                                    neg_h += 1
                                if 'BL' in line or 'FU' in line:
                                    neg_s += 1
                        neg_content = [f'NEG_{num_neg}', neg_h, neg_s, round(neg_h / neg_s, 3),
                                       round(neg_h / (neg_h + neg_s), 3)]
                        neg_temp.append((f'NEG_{num_neg}', neg_h, neg_s, round(neg_h / neg_s, 3),
                                         round(neg_h / (neg_h + neg_s), 3)))
                        ratio_per_cluster_df.loc[jx] = neg_content

                    if 'pos' in file:  # POS
                        num_pos += 1
                        pos_h = 0
                        pos_s = 0
                        for line in lines:
                            if line.startswith('>'):
                                if 'HD' in line:
                                    pos_h += 1
                                if 'BL' in line or 'FU' in line:
                                    pos_s += 1
                        pos_content = [f'POS_{num_pos}', pos_h, pos_s, round(pos_h / pos_s, 3),
                                       round(pos_h / (pos_h + pos_s), 3)]
                        pos_temp.append((f'POS_{num_pos}', pos_h, pos_s, round(pos_h / pos_s, 3),
                                         round(pos_h / (pos_h + pos_s), 3)))
                        ratio_per_cluster_df.loc[jx] = pos_content

        # print('H:', neg_temp)
        # print('S:', pos_temp)
        list_of_df.append((model, ratio_per_cluster_df))
    for name, df in list_of_df:
        print(name + '\\\\')
        df.sort_values(by=['#HEALTHY'], inplace=True, ascending=False)
        print(df.to_latex(index=False))
        print('\\\\')


compute_h_d_ratio()