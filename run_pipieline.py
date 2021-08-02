import time
import pandas as pd
import joblib
import matplotlib.pyplot as plt

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

    def set_embed(self, spread=1.0, min_dist=0.1, a=1.0, b=1.0):

        self.data.set_data_frame()
        self.data.set_embedding(spread=spread, min_dist=min_dist, a=a, b=b)
        print('Embedding check.')

    def set_graph(self):

        self.data.set_graph()
        print('Graph check.')

    def find_cluster(self, gamma=1.065):

        self.data.gamma = gamma
        self.data.calculate_communities(save_cluster=False)
        # self.data.plot_cluster(save_fig=False)
        print('', self.data.plot_location)

    def set_cluster(self):
        self.data.plot_cluster()

    def build_feature(self):

        print('Building feature vector ...')
        self.data.set_response()
        self.data.calculate_feature_vector()
        print('check')
        # print(BLHD.feature_vector)

    def sk_build_model(self, c=0.01):

        self.data.regularization_c = c
        self.data.sk_build_model()

    def sm_build_model(self, solver, concat):

        self.data.sm_build_model(solver, concat=concat)

    def sm_build_reg_model(self, l1_w):

        self.data.sm_build_reg_model(l1_w)


    def sm_predict_and_score(self):
        self.data.sm_eval()


    def print_confusion_matrix(self):

        self.data.print_confusion_matrix()

    def print_score(self):

        self.data.sk_model_evaluation()
        print(self.data.model_score)

    def print_report(self):

        self.data.sk_set_model_report()
        print(self.data.model_report)

    def plot_roc(self):

        self.data.sk_plot_roc()

    def highlight_cluster(self):

        self.data.sk_prepare_traceback()
        self.data.sk_set_z()
        self.data.sk_highlight_clusters()

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
    path_to_blhd_identity_dm = '/home/ubuntu/Enno/mnt/volume/distance_matrices/BLHD_ID'
    path_to_blhd_b45_dm = '/home/ubuntu/Enno/mnt/volume/distance_matrices/BLHD_BLOSUM45_10_0.5_DM'
    path_to_blhd_b62_dm = '/home/ubuntu/Enno/mnt/volume/distance_matrices/BLHD_BLOSUM62_10_0.5_DM'
    path_to_blhd_b80_dm = '/home/ubuntu/Enno/mnt/volume/distance_matrices/BLHD_BLOSUM80_10_0.5_DM'
    path_to_blhd_gonnet_dm = '/home/ubuntu/Enno/mnt/volume/distance_matrices/BLHD_GONNET1992_10_0.5_DM'
    path_to_blhd_clustalo_dm = '/home/ubuntu/Enno/mnt/volume/distance_matrices/BLHD_CLUSTALO'

    PATHs = [path_to_blhd_identity_dm, path_to_blhd_b45_dm, path_to_blhd_b62_dm, path_to_blhd_b80_dm, path_to_blhd_gonnet_dm]

    path = path_to_blhd_gonnet_dm

    GN_GAMMAS = [1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.11, 1.12, 1.13, 1.14]
    CL_GAMMAS = [1.000, 1.000, 1.001, 1.001, 1.002, 1.002, 1.003, 1.003]

    BLHD = DoStuff()

    for p in PATHs:
        path = p
        obj_props = path.split(sep='/')[-1]
        print('CURRENT RUN:', obj_props, '\n')

        # Initialize run


        BLHD.set_dm_parameter()
        BLHD.set_directories(dm_path=path)
        BLHD.set_sequence_info()
        BLHD.set_dm()
        BLHD.set_graph()
        BLHD.set_embed()

        for gamma in GN_GAMMAS:
            print('GAMMA:', gamma, '\n')
            BLHD.find_cluster(gamma)

            BLHD.build_feature()
            # BLHD.data.plot_sequence_distribution()

            BLHD.sm_build_model(solver='lbfgs', concat=False)
            BLHD.sm_predict_and_score()
            BLHD.data.sk_cv_scores()
            # BLHD.data.sm_feature_plot()

            BLHD.data.append_cluster_of_interest()
            BLHD.data.compute_cluster_overlap()
            BLHD.data.performance_sampling()

            # BLHD.sm_build_model(solver='newton', concat=False)
            # BLHD.sm_predict_and_score()
            # BLHD.data.sk_cv_scores()
            # BLHD.data.sm_feature_plot()

        # BLHD.data.plot_performance(obj_props)

    """CL_CONC_GAMMAS = [1.001, 1.003]
    BL_CONC_GAMMAS = [1.05, 1.065]
    BLHD.data.concatenate_features(CL_CONC_GAMMAS)

    BLHD.sm_build_model(solver='lbfgs', concat=False)
    BLHD.sm_predict_and_score()
    BLHD.data.sk_cv_scores()

    BLHD.sm_build_model(solver='newton', concat=False)
    BLHD.sm_predict_and_score()
    BLHD.data.sk_cv_scores()
"""
    # # # # # PARAMETER INFO # # # # # # # # # #
        # spread    =    [0.1,      10]            #  used to control the inter-cluster distances
        #   -> a, b become smaller                 #
        # min_dist  =    [0.0001,    2]            #
        #   -> a becomes smaller                   #  used to conrtol the size of the clusters
        #   -> b becomes bigger                    #
        # a         =    [0.0001, (0.1, 10)  100]  #
        # b         =    [0.1,     2,5]            #
        # # # # # # # # # # # # # # # # # # # #  # #



        # l1_weights = [0, 0.2, 0.4, 0.6, 0.8, 1]



    # ‘newton’ for Newton - Raphson, ‘nm’ for Nelder-Mead
    # ‘bfgs’ for Broyden - Fletcher - Goldfarb - Shanno(BFGS)
    # ‘lbfgs’ for limited - memory BFGS with optional box constraints
    # ‘powell’ for modified Powell’s method
    # ‘cg’ for conjugate gradient
    # ‘ncg’ for Newton - conjugate gradient
    # ‘basinhopping’ for global basin-hopping solver


    """for gamma in gamma_batch:
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

    """
        A = [0.1, 0.25, 0.5, 1, 2.5, 5, 10]
        B = [0.1, 0.25, 0.5, 1, 1.5, 2, 2.5]
        for a in A:
            for b in B:
                print('A:', a, 'B:', b)
                BLHD.set_embed()
                BLHD.set_cluster()
        """
    """for path in PATHs:
        # path = path_to_clustalo_dm
        obj_props = path.split(sep='/')[-1]
        print('CURRENT RUN:', obj_props, '\n')

        # Initialize run
        BLHD = DoStuff()

        BLHD.set_dm_parameter()
        BLHD.set_directories(dm_path=path)
        BLHD.set_sequence_info()
        BLHD.set_dm()
        BLHD.set_graph()
        BLHD.set_embed()


        save_path = "/home/ubuntu/Enno/mnt/volume/objects/%s" % obj_props
        joblib.dump(BLHD.data.embedding, save_path)

        inc = 0.005
        GAMMAs = [inc*i for i in range(200, 241)]
        BL_GAMMAS = [1.0, 1.075, 1.1, 1.115] # [] #1.025, 1.075, 1.100]
        # BLHD.data.concatenate_features(GAMMAS)

        for gamma in BL_GAMMAS:
            print('GAMMA:', gamma, '\n')
            BLHD.find_cluster(gamma)

            BLHD.build_feature()
            # BLHD.data.plot_sequence_distribution()
            BLHD.sm_build_model(solver='lbfgs', concat=False)
            BLHD.sm_build_model(solver='newton', concat=False)
            BLHD.sm_predict_and_score()
            BLHD.data.sk_cross_val_score()

        BL_CONC_GAMMAS = [1.045, 1.07]
        BLHD.data.concatenate_features(BL_CONC_GAMMAS)

        BLHD.sm_build_model(solver='lbfgs', concat=False)
        BLHD.sm_build_model(solver='newton', concat=False)
        BLHD.sm_predict_and_score()
        BLHD.data.sk_cross_val_score()"""
