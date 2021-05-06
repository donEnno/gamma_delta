import os
import pandas as pd
from pip._vendor.urllib3.connectionpool import xrange


path = '/Users/Enno/PycharmProjects/gamma_delta/data/HD_VDJTools'

# Converts VDJ.txt - files to pandas df
def convert_to_df(p):

    all_files = os.listdir(p)
    output = []

    for file in all_files:
        with open(p + '/' + file, 'rt') as fd:
            output.append(pd.read_csv(fd, sep='\t', header=0))
    return output


# Read the BLOSUM50 table
def read_BLOSUM(fname):
    d = {}
    lines = open(fname, "rt").readlines()
    alpha = lines[0].rstrip('\n\r').split()
    assert(len(alpha) == len(lines)-1)
    for r in lines[1:]:
        r = r.rstrip('\n\r').split()
        a1 = r[0]
        for a2, score in zip(alpha, r[1:]):
            d[(a1, a2)] = int(score)
    return d


# Needleman-Wunsch algorithm as described in wikipedia
def needleman_wunsch(seq1, seq2, b, g):
    """

    :param seq1:
    :param seq2:
    :param b: Blossum Matrix
    :param g: Gap Penalty
    :return M: Scoring Matrix
    """
    m = len(seq1) + 1  # Rows
    n = len(seq2) + 1  # Columns

    M = create_matrix(m, n)

    for i in range(0, m):
        M[i][0] = i * g
    for j in range(0, n):
        M[0][j] = j * g

    for i in range(1, m):
        for j in range(1, n):
            match = M[i - 1][j - 1] + b[(seq1[i - 1], seq2[j - 1])]
            delete = M[i - 1][j] + g
            insert = M[i][j - 1] + g
            M[i][j] = max(match, delete, insert)

    return M


# Create an empty matrix
def create_matrix(m, n):
    return [[0] * n for _ in xrange(m)]  # xrange uses less space than range (?)


# Calculate the scoring matrix between organisms
def scoring_matrix(patient, b, g):
    """
    :param patient: Patient of Interest
    :param b: Blossum Matrix
    :param g: Gap Penalty
    :return scoring_matrix: Returns Matrix with all pairwise alignment scores for patient
    """
    score_matrix = create_matrix(len(patient), len(patient))
    d = read_BLOSUM(b)

    i = 0
    for x in patient['cdr3aa']:
        j = 0
        for y in patient['cdr3aa']:
            M = needleman_wunsch(x, y, d, g)
            score_matrix[i][j] = (M[-1][-1], i, j)
            j = j + 1
        # print(i)
        i = i + 1

    return score_matrix


def calculate_pairwise_alignment(seq_1, seq_2, b, g):
    """
    :param seq_1: Str or List
    :param seq_2:
    :param b: Blossum Matrix
    :param g: Gap Penalty
    :return: Calculates either a pairwise alignment (type(seq_1) = str) or calculates pairwise alignement between
             first element of seq_1 (type = list) and applies all changes to all further elements of seq_1
    """

    if type(seq_1) == str:
        n = len(seq_1)-1
    else:
        n = len(seq_1[0]) - 1

    m = len(seq_2)-1
    d = read_BLOSUM(b)

    if type(seq_1) == str:
        sequence = ""
    else:
        sequences = ['' for i in range(len(seq_1))]
    alignment = ""

    if type(seq_1) == str:
        sm = needleman_wunsch(seq_1, seq_2, d, g)
    else:
        sm = needleman_wunsch(seq_1[0], seq_2, d, g)

    # for line in sm:
    #     print(line)

    while n > 0 and m > 0:
        # print(n)
        # print(m)

        replace = sm[n-1][m-1]
        insert = sm[n][m-1]
        delete = sm[n-1][m]
        best_option = max(replace, insert, delete)

        if replace == best_option:
            if type(seq_1) == str:
                sequence = seq_1[n] + sequence
            else:
                for i in range(len(seq_1)):
                    sequences[i] = seq_1[i][n] + sequences[i]
            alignment = seq_2[m] + alignment
            n = n - 1
            m = m - 1

        elif insert == best_option:
            if type(seq_1) == str:
                sequence = "*" + sequence
            else:
                for i in range(len(seq_1)):
                    sequences[i] = '*' + sequences[i]
            alignment = seq_2[m] + alignment
            m = m - 1

        elif delete == best_option:
            if type(seq_1) == str:
                sequence = seq_1[n] + sequence
            else:
                for i in range(len(seq_1)):
                    sequences[i] = seq_1[i][n] + sequences[i]
            alignment = "*" + alignment
            n = n - 1
        else:
            return "f"

    if n == 0 and m == 0:
        if type(seq_1) == str:
            sequence = seq_1[n] + sequence
        else:
            for i in range(len(seq_1)):
                sequences[i] = seq_1[i][n] + sequences[i]
        alignment = seq_2[m] + alignment
    elif n == 0 and m > 0:
        if type(seq_1) == str:
            sequence = "*" + sequence
        else:
            for i in range(len(seq_1)):
                sequences[i] = '*' + sequences[i]
        alignment = seq_2[m] + alignment
    elif m == 0 and n > 0:
        if type(seq_1) == str:
            sequence = seq_1[n] + sequence
        else:
            for i in range(len(seq_1)):
                sequences[i] = seq_1[i][n] + sequences[i]
        alignment = "*" + alignment
    else:
        return "f"

    if type(seq_1) == str:
        return [sequence, alignment]
    else:
        sequences.append(alignment)
        return sequences


def calculate_pairwise_alignment_v2(seq_1, seq_2, b, g):
    """
    :param seq_1: List
    :param seq_2:
    :param b: Blossum Matrix
    :param g: Gap Penalty
    :return: Calculates either a pairwise alignment (type(seq_1) = str) or calculates pairwise alignement between
             first element of seq_1 (type = list) and applies all changes to all further elements of seq_1
    """

    if type(seq_1) == str:
        n = len(seq_1)-1
    else:
        n = len(seq_1[0]) - 1

    m = len(seq_2)-1
    d = read_BLOSUM(b)

    if type(seq_1) == str:
        sequence = ""
    else:
        sequences = ['' for i in range(len(seq_1))]
    alignment = ""

    if type(seq_1) == str:
        sm = needleman_wunsch(seq_1, seq_2, d, g)
    else:
        sm = needleman_wunsch(seq_1[0], seq_2, d, g)

    # for line in sm:
    #     print(line)

    while n > 0 and m > 0:
        # print(n)
        # print(m)

        replace = sm[n-1][m-1]
        insert = sm[n][m-1]
        delete = sm[n-1][m]
        best_option = max(replace, insert, delete)

        if replace == best_option:
            if type(seq_1) == str:
                sequence = seq_1[n] + sequence
            else:
                for i in range(len(seq_1)):
                    sequences[i] = seq_1[i][n] + sequences[i]
            alignment = seq_2[m] + alignment
            n = n - 1
            m = m - 1

        elif insert == best_option:
            if type(seq_1) == str:
                sequence = "*" + sequence
            else:
                for i in range(len(seq_1)):
                    sequences[i] = '*' + sequences[i]
            alignment = seq_2[m] + alignment
            m = m - 1

        elif delete == best_option:
            if type(seq_1) == str:
                sequence = seq_1[n] + sequence
            else:
                for i in range(len(seq_1)):
                    sequences[i] = seq_1[i][n] + sequences[i]
            alignment = "*" + alignment
            n = n - 1
        else:
            return "f"

    if n == 0 and m == 0:
        if type(seq_1) == str:
            sequence = seq_1[n] + sequence
        else:
            for i in range(len(seq_1)):
                sequences[i] = seq_1[i][n] + sequences[i]
        alignment = seq_2[m] + alignment
    elif n == 0 and m > 0:
        if type(seq_1) == str:
            sequence = "*" + sequence
        else:
            for i in range(len(seq_1)):
                sequences[i] = '*' + sequences[i]
        alignment = seq_2[m] + alignment
    elif m == 0 and n > 0:
        if type(seq_1) == str:
            sequence = seq_1[n] + sequence
        else:
            for i in range(len(seq_1)):
                sequences[i] = seq_1[i][n] + sequences[i]
        alignment = "*" + alignment
    else:
        return "f"

    if type(seq_1) == str:
        return [sequence, alignment]
    else:
        sequences.append(alignment)
        return sequences


# Makes matrix M a triangular matrix. Sets the upper triangle to (0, 0, 0).
def triangulate_matrix(m):

    for i in range(len(m)):
        for j in range(len(m[0])):
            if j >= i:
                m[i][j] = (0, 0, 0)

    return m


# Returns maximum of a matrix whose entries are in (value, #seq_1, #seq_2)-format.
def get_matrix_max(m):

    max_value = None

    for i in range(len(m)):
        for j in range(len(m[0])):
            if max_value is None:
                max_value = m[i][j]
            if max_value[0] <= m[i][j][0]:
                max_value = m[i][j]

    return max_value


def calculate_mulitple_alignment(df, sm):
    """
    :param df: Patient of interest.
    :param sm: Scoring Matrix from scoring_matrix() methode
    :return: Prints multiple alignment.
    """

    # Later output.
    set_of_aligned_seq = []
    # Keeps track of which seqs are already in the MSA
    ID_done = []
    c = 0

    score_mat = triangulate_matrix(sm)
    current = get_matrix_max(score_mat)
    max_score, ID_seq1, ID_seq2 = current[0], current[1], current[2]

    BLSM50, BLSM80 = 'needle/blosum50.txt', 'needle/blosum80.txt'

    set_of_aligned_seq.extend(calculate_pairwise_alignment(df['cdr3aa'][ID_seq1], df['cdr3aa'][ID_seq2], BLSM50, -9))

    ID_done.append(ID_seq1)
    ID_done.append(ID_seq2)

    # Avoid finding the same maxima again.
    sm[ID_seq1][ID_seq2] = (0, 0, 0)            # Only one needed.
    sm[ID_seq2][ID_seq1] = (0, 0, 0)            # Don't know yet which one.

    while c <= len(sm):
        # print(c)
        # Reset.
        current = (0, 0, 0)
        max_score = 0

        # Find maximum again ...
        for x in ID_done:
            # Horizontal
            for y in sm[x]:
                if y[1] in ID_done and y[2] in ID_done:
                    continue
                if y[0] > max_score:
                    max_score = y[0]
                    current = y

            for i in range(len(sm)):
                y = sm[i][x]
                if y[1] in ID_done and y[2] in ID_done:
                    continue
                if y[0] > max_score:
                    max_score = y[0]
                    current = y

        ID_seq1 = current[1]
        ID_seq2 = current[2]

        if ID_seq1 not in ID_done:
            temp = calculate_pairwise_alignment(set_of_aligned_seq, df['cdr3aa'][ID_seq2], BLSM50, -5)
            set_of_aligned_seq = temp
            ID_done.append(ID_seq1)
        if ID_seq2 not in ID_done:
            temp = calculate_pairwise_alignment(set_of_aligned_seq, df['cdr3aa'][ID_seq1], BLSM50, -5)
            set_of_aligned_seq = temp
            ID_done.append(ID_seq2)

        sm[ID_seq1][ID_seq2] = (0, 0, 0)
        sm[ID_seq2][ID_seq1] = (0, 0, 0)

        c += 1
    # print(ID_done)
    for line in set_of_aligned_seq:
        print(line)


# Convert all patients to df
all_patients = convert_to_df(path)

patient1 = all_patients[0]
patient2 = all_patients[1]

# Cut patient 1 short for testing purpose
# patient1 = patient1[:10]

sm_patient1 = scoring_matrix(patient1, 'needle/blosum50.txt', -5)
# print('Matrix done.')

calculate_mulitple_alignment(patient1, sm_patient1)


"""
For further use when it comes to dendrograms.

# Calculate the distance scoring matrix
def scoring_distance_matrix(scoring_matrix):
    scoring_distance_matrix = create_matrix(len(scoring_matrix[0]), len(scoring_matrix[0]))
    maxR = get_matrix_max(scoring_matrix)

    for i in range(0, len(organisms_table)):
        for j in range(0, len(organisms_table)):
            scoring_distance_matrix[i][j] = abs(scoring_matrix[i][j] - maxR)

    return scoring_distance_matrix


# Return the max value in a matrix, used in
# scoring_distance_matrix method
def get_matrix_max(matrix):
    max_value = None

    for i in range(0, len(matrix[0])):
        for j in range(0, len(matrix[0])):
            if (max_value == None):
                max_value = matrix[i][j]
            if (matrix[i][j] >= max_value):
                max_value = matrix[i][j]

    return max_value
"""
