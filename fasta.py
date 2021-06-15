import joblib
import pandas as pd
import os
import numpy as np


def txt_to_fasta(mode: str, one_file=False):

    path = '/home/ubuntu/Enno/gammaDelta/data/'
    pre = ''

    if mode == 'HD':
        path = path + 'HD_txt/'
        pre = 'HD_'
    if mode == 'BL':
        path = path + 'BL_txt/'
        pre = 'BL_'
    if mode == 'FU':
        path = path + 'FU_txt/'
        pre = 'FU_'

    list_of_txt = os.listdir(path)
    list_of_txt.sort()

    list_of_dataframe = []

    for file in list_of_txt:
        with open(path + file, 'rt') as fd:
            list_of_dataframe.append(pd.read_csv(fd, sep='\t', header=0))

    if one_file:
        with open('/home/ubuntu/Enno/gammaDelta/sequence_data/' + f'{mode}_fasta' + f'/{pre}ALL_SEQUENCES.fasta', 'w') as wf:
            i = 1
            for patient in list_of_dataframe:
                j = 1
                for sequence in patient['cdr3aa']:
                    wf.writelines('>'+pre+'PATIENT_'+str(i)+'_SEQUENCE_'+str(j) + '\n' + sequence + '\n')
                    j += 1
                i += 1

    else:
        i = 1
        for patient in list_of_dataframe:
            j = 1
            with open('/home/ubuntu/Enno/gammaDelta/sequence_data/' + f'{mode}_fasta' + f'/{pre}PATIENT_{i}.fasta', 'w') as wf:
                for sequence in patient['cdr3aa']:
                    wf.writelines('>' + pre + 'PATIENT_' + str(i) + '_SEQUENCE_' + str(j) + '\n' + sequence + '\n')
                    j += 1
                i += 1


def ugly_combine_hd_bl():

    path = '/home/ubuntu/Enno/gammaDelta/data/'
    pre = 'BL_'

    list_of_txt_bl = os.listdir(path + 'BL_txt/')
    list_of_txt_bl.sort()

    list_of_txt_hd = os.listdir(path + 'HD_txt/')
    list_of_txt_hd.sort()

    list_of_dataframe = []

    for file in list_of_txt_bl:
        with open(path + 'BL_txt/' + file, 'rt') as fd:
            list_of_dataframe.append(pd.read_csv(fd, sep='\t', header=0))

    for file in list_of_txt_hd:
        with open(path + 'HD_txt/' + file, 'rt') as fd:
            list_of_dataframe.append(pd.read_csv(fd, sep='\t', header=0))

    print(len(list_of_dataframe))

    with open('/home/ubuntu/Enno/gammaDelta/sequence_data/' + f'BLHD_fasta' + f'/BLHD_ALL_SEQUENCES.fasta',
              'w') as wf:
        i = 1
        for patient in list_of_dataframe:
            if i == 67:
                i = 1
                pre = 'HD_'
            j = 1
            for sequence in patient['cdr3aa']:
                wf.writelines('>' + pre + 'PATIENT_' + str(i) + '_SEQUENCE_' + str(j) + '\n' + sequence + '\n')
                j += 1
            i += 1


if __name__ == '__main__':
    print('Let\'s go!')

    co_to_np = False
    if co_to_np:
        with open(r"/home/ubuntu/Enno/mnt/volume/distance_matrices/CO_BLHD_DM.faa.mat", 'rt') as dm:

            mat = np.zeros((29594, 29594))
            ix = 0
            lines = dm.readlines()[1:]

            for row in lines:
                row = row.split()[1:]
                mat[ix] = row
                ix += 1
                if ix % 1000 == 0:
                    print(ix)

            print(mat.shape)
            joblib.dump(mat, "/home/ubuntu/Enno/mnt/volume/distance_matrices/NP_BLHD_DM")



