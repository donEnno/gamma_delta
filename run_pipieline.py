from pipeline import Data


class DoStuff:
    def __init__(self):
        self.data = Data()
        """ Initialize
        print('Initializing ...')
        BLHD = Data()
        print('check')
        """

    def set_dm_parameter(self,
                         origin='BLHD',
                         substitution_matrix='BLOSUM45',
                         gap_open=10,
                         gap_extend=0.5):

        print('Alignment properties ...')
        self.data.origin = origin
        self.data.substitution_matrix = substitution_matrix
        self.data.gap_open = gap_open
        self.data.gap_extend = gap_extend
        print('check')

    def set_directories(self, dm_path):

        print('Directories ...')
        self.data.fasta_location = '/home/ubuntu/Enno/gammaDelta/sequence_data/BLHD_fasta/BLHD_ALL_SEQUENCES.fasta'
        self.data.dm_location = dm_path
        self.data.plot_location = '/home/ubuntu/Enno/gammaDelta/plots/BLHD_TEST_1.png'
        self.data.cluster_location = self.data.set_cluster_location()
        print('check')

    def set_sequence_info(self):

        self.data.set_sequences()
        self.data.set_num_seq()
        print('Number of sequences: ', self.data.number_of_sequences)

    def set_dm(self):

        self.data.set_dm()

    def set_embed(self):

        self.data.set_data_frame()
        self.data.set_embedding()
        print('Embedding check.')

    def set_graph(self):

        self.data.set_graph()
        print('Graph check.')

    def find_cluster(self, gamma=1.065):

        self.data.gamma = gamma
        self.data.calculate_communities(save_cluster=False)
        self.data.plot_cluster(save_fig=True)
        print('', self.data.plot_location)

    def build_feature(self):

        print('Building feature vector ...')
        self.data.calculate_feature_vector()
        self.data.set_response()
        print('check')
        # print(BLHD.feature_vector)

    def build_model(self, c=0.01):

        self.data.regularization_c = c
        self.data.build_model()

    def print_confusion_matrix(self):

        self.data.print_confusion_matrix()

    def print_score(self):

        self.data.model_evaluation()
        print(self.data.model_score)

    def print_report(self):

        self.data.set_model_report()
        print(self.data.model_report)

    def plot_roc(self):

        self.data.plot_roc()

    def highlight_cluster(self):

        self.data.prepare_traceback()
        self.data.set_z()
        self.data.highlight_clusters()

    def calculate_distance_matrices(self, sms):
        self.set_dm_parameter(origin='BLHD', gap_open=10, gap_extend=0.5)
        self.data.fasta_location = '/home/ubuntu/Enno/gammaDelta/sequence_data/BLHD_fasta/BLHD_ALL_SEQUENCES.fasta'

        self.set_sequence_info()

        for substitution_matrix in sms:

            self.data.substitution_matrix = substitution_matrix
            self.data.dm_location = fr'/home/ubuntu/Enno/mnt/volume/distance_matrices/' + self.data.origin + '_' + self.data.substitution_matrix + '_' + str(self.data.gap_open) + '_' + str(self.data.gap_extend) + '_DM'
            print(self.data.dm_location)
            self.data.calculate_distance_matrix()


if __name__ == '__main__':
    """path_to_identity_dm = '/home/ubuntu/Enno/mnt/volume/distance_matrices/TEST'
    path_to_clustalo_dm = '/home/ubuntu/Enno/mnt/volume/distance_matrices/NP_BLHD_DM'
    BLHD = DoStuff()

    BLHD.set_dm_parameter()
    BLHD.set_directories(dm_path=path_to_identity_dm)
    BLHD.set_sequence_info()
    BLHD.set_dm()

    BLHD.set_embed()
    BLHD.set_graph()

    gamma_batch = [1.05, 1.055, 1.06, 1.065, 1.07, 1.075]
    c_batch = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    for gamma in gamma_batch:
        print('GAMMA', gamma)
        BLHD.find_cluster(gamma=gamma)
        BLHD.build_feature()
        for c in c_batch:

            print('C', c)
            BLHD.build_model(c=c)
            BLHD.print_confusion_matrix()
            BLHD.plot_roc()
            BLHD.highlight_cluster()
"""
    BLHD = DoStuff()
    sms = ['BLOSUM45', 'BLOSUM62', 'BLOSUM80', 'GONNET1992']
    BLHD.calculate_distance_matrices(sms=sms)
