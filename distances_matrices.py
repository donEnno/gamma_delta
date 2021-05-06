import time
from Bio import SeqIO, pairwise2, Align
from Bio.Align import substitution_matrices
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

"""
* +++++++++ *
| pairwise2 |
* +++++++++ *

water (local), needle (global)
pairwise2.align.globalxx(seq1.seq, seq2.seq)
                      xx = only matches are counted
                      
blosum62 = substitution_matrices.load("BLOSUM62")
alignments = pairwise2.align.globalds(seq1.seq, seq2.seq, blosum62, -10 GO, -0.5 GE)

pairwise2.align.localms("AGAACT", "GAC", 5, -4, -2, -0.5)
                                        m   mm  go   ge

enhance speed: score_only=True or one_alignment_only = True

blosum62 = substitution_matrices.load("BLOSUM62")


alignments = pairwise2.align.globalds(seqA, seqB, blosum62, -10, -0.5)

print(pairwise2.format_alignment(*alignments[0]))

['BENNER22', 'BENNER6', 'BENNER74', 'BLOSUM45', 'BLOSUM50', 'BLOSUM62', 'BLOSUM80', 'BLOSUM90', 'DAYHOFF', 'FENG', 
'GENETIC', 'GONNET1992', 'HOXD70', 'JOHNSON', 'JONES', 'LEVIN', 'MCLACHLAN', 'MDM78', 'NUC.4.4', 'PAM250', 'PAM30', 
 'PAM70', 'RAO', 'RISLER', 'SCHNEIDER', 'STR', 'TRANS']

"""

data_dir = r'C:\Users\Enno\PycharmProjects\gamma_delta\data\sequences'


def get_num_seq(f):
    num = len([1 for line in open(f) if line.startswith(">")])
    return num


def fasta_to_distance_matrix(patient, mat):
    """
    :param patient:
    :param mat: substitutionmatrix:
    :return:
    """
    file = fr'C:\Users\Enno\PycharmProjects\gamma_delta\data\sequences\patient_{patient}.fasta'
    matrix = substitution_matrices.load(mat)

    n = get_num_seq(file)
    dm = np.zeros((n, n))

    i = 0
    j = 0

    t0 = time.time()
    for seqA in SeqIO.parse(file, "fasta"):
        print("Currently working on sequence ", i)
        for seqB in SeqIO.parse(file, "fasta"):
            if i <= j:
                continue

            dm[i][j] = pairwise2.align.globalds(seqA.seq, seqB.seq, matrix, -10, -0.5, score_only=True)
            j += 1
            # print(p1_dm)
        j = 0
        i += 1
    np.savetxt(f'P{patient}_DM_{mat}.csv', dm, delimiter=',')
    t1 = time.time()-t0
    print("Time elapsed: ", t1)


def calculate_substitution_matrices(patients, matrices):
    for sm in sm_batch:
        print("Current substitution matrix: ", sm)
        for p in p_batch:
            print("Working on patient ", str(p))
            fasta_to_distance_matrix(p, sm)


sm_batch = ["BLOSUM45", "BLOSUM80"]
p_batch = [1, 3, 13, 15, 17, 18, 19, 20, 21, 28, 29]


p3_dm_b45 = np.loadtxt('P3_DM_BLOSUM45.csv', delimiter=',')
p3_dm_b62 = np.loadtxt('P3_DM_BLOSUM62.csv', delimiter=',')
p3_dm_b89 = np.loadtxt('P3_DM_BLOSUM80.csv', delimiter=',')
p3_dm_gonnet = np.loadtxt('P3_DM_GONNET1992.csv', delimiter=',')

g_p3_dm_b45 = nx.from_numpy_matrix(p3_dm_b45)
g_p3_dm_b62 = nx.from_numpy_matrix(p3_dm_b62)
g_p3_dm_b80 = nx.from_numpy_matrix(p3_dm_b89)
g_p3_dm_gonnet = nx.from_numpy_matrix(p3_dm_gonnet)

# nx.draw(g_p3_dm_b45, with_labels=True)
# plt.savefig('g_p3_dm_b45.png', dpi=500)
# plt.show()
#
# nx.draw(g_p3_dm_b62, with_labels=True)
# plt.savefig('g_p3_dm_b62.png', dpi=500)
# plt.show()
#
# nx.draw(g_p3_dm_b80, with_labels=True)
# plt.savefig('g_p3_dm_b80.png', dpi=500)
# plt.show()
#
# nx.draw(g_p3_dm_gonnet, with_labels=True)
# plt.savefig('g_p3_dm_gonnet.png', dpi=500)
# plt.show()

g_p3_dm_b45_spring = nx.spring_layout(g_p3_dm_b45)
nx.draw_networkx_nodes(g_p3_dm_b45, g_p3_dm_b45_spring, cmap=plt.get_cmap('jet'), node_size=300)
nx.draw_networkx_labels(g_p3_dm_b45, g_p3_dm_b45_spring,)
nx.draw_networkx_edges(g_p3_dm_b45, g_p3_dm_b45_spring, arrows=True)
plt.savefig('g_p3_dm_b45_spring.png', dpi=500)

g_p3_dm_b62_spring = nx.spring_layout(g_p3_dm_b62)
nx.draw_networkx_nodes(g_p3_dm_b62, g_p3_dm_b62_spring, cmap=plt.get_cmap('jet'), node_size=300)
nx.draw_networkx_labels(g_p3_dm_b62, g_p3_dm_b62_spring,)
nx.draw_networkx_edges(g_p3_dm_b62, g_p3_dm_b62_spring, arrows=True)
plt.savefig('g_p3_dm_b62_spring.png', dpi=500)

g_p3_dm_b80_spring = nx.spring_layout(g_p3_dm_b80)
nx.draw_networkx_nodes(g_p3_dm_b80, g_p3_dm_b80_spring, cmap=plt.get_cmap('jet'), node_size=300)
nx.draw_networkx_labels(g_p3_dm_b80, g_p3_dm_b80_spring,)
nx.draw_networkx_edges(g_p3_dm_b80, g_p3_dm_b80_spring, arrows=True)
plt.savefig('g_p3_dm_b80_spring.png', dpi=500)

g_p3_dm_gonnet_spring = nx.spring_layout(g_p3_dm_gonnet)
nx.draw_networkx_nodes(g_p3_dm_gonnet, g_p3_dm_gonnet_spring, cmap=plt.get_cmap('jet'), node_size=300)
nx.draw_networkx_labels(g_p3_dm_gonnet, g_p3_dm_gonnet_spring,)
nx.draw_networkx_edges(g_p3_dm_gonnet, g_p3_dm_gonnet_spring, arrows=True)
plt.savefig('g_p3_dm_gonnet_spring.png', dpi=500)

