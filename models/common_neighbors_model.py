from models.base_model import BaseGameRecommendationModel, SAVED_MODELS_PATH
import networkx as nx
import pickle
import sys
import os
import scipy
import numpy as np
from scipy.sparse import coo_matrix

class CommonNeighbors(BaseGameRecommendationModel):
    def __init__(self, path_length_2_weight = 1, path_length_3_weight = 1):
        super().__init__()
        self.path_length_2_weight = path_length_2_weight
        self.path_length_3_weight = path_length_3_weight

    def name(self):
        return 'common_neighbors'

    def get_user_game_adjacency_matrix(self, users_games_df):
        user_indices = users_games_df['user_id'].apply(lambda id: self.node_to_index[id]).tolist()
        game_indices = users_games_df['game_id'].apply(lambda id: self.node_to_index[id]).tolist()
        scores = users_games_df['score'].tolist()
        rows = user_indices + game_indices
        cols = game_indices + user_indices
        data = scores + scores

        sparse_matrix = coo_matrix((data, (rows, cols)), shape=(len(self.index_to_node), len(self.index_to_node)))
        sparse_matrix_csr = sparse_matrix.tocsr()
        return sparse_matrix_csr

    # TODO Set different interaction strengths based on game ownership, user liked, and user disliked
    def train(self):
        # TODO train on downloaded interactions
        assert self.data_loader.cache_local_dataset, 'Method requires full load.'
        self.index_to_node = self.data_loader.get_all_node_ids()
        self.node_to_index = {node: ii for ii, node in enumerate(self.index_to_node)}
        self.game_nodes = self.data_loader.get_game_node_ids()

        train_users_games_df = self.data_loader.users_games_df[self.data_loader.users_games_df['data_split'] == 'train']
        self.matrix = self.get_user_game_adjacency_matrix(train_users_games_df)

    def _fine_tune(self, user_id, new_user_games_df, new_interactions_df, all_user_games_df, all_interactions_df):
        if not user_id in self.node_to_index:
            user_connections = np.full((self.matrix.shape[0]), 0)
        else:
            user_connections = self.matrix.getrow(self.node_to_index[user_id]).toarray().flatten()
        for ii, row in new_user_games_df.iterrows():
            if row['game_id'] in self.node_to_index:
                user_connections[self.node_to_index[row['game_id']]] = row['score']
        for ii, row in new_interactions_df.iterrows():
            if row['game_id'] in self.node_to_index:
                user_connections[self.node_to_index[row['game_id']]] = row['score']
        if not user_id in self.node_to_index:
            self.node_to_index[user_id] = len(self.index_to_node)
            self.index_to_node.append(user_id)
            self.matrix = scipy.sparse.vstack([self.matrix, user_connections.reshape(1, -1)])
            user_connections = np.append(user_connections, 0)
            self.matrix = scipy.sparse.hstack([self.matrix, user_connections.reshape(-1, 1)])
            self.matrix = self.matrix.tocsr() # Have to convert because we are doing an hstack on all columns.
        else:
            indices = np.where(user_connections != 0)[0]
            self.matrix[self.node_to_index[user_id], indices] = user_connections[indices]
            self.matrix[indices, self.node_to_index[user_id]] = user_connections[indices]

    def get_score_between_user_and_game(self, user, game):
        user_index = self.node_to_index[user]
        game_index = self.node_to_index[game]
        return self.path_length_2_weight * (self.matrix[user_index, :] @ self.matrix[:, game_index])[0, 0] + self.path_length_3_weight * (self.matrix[user_index, :] @ self.matrix @ self.matrix[:, game_index])[0, 0]

    def score_and_predict_n_games_for_user(self, user, N=None, should_sort=True):
        games_to_filter_out = self.data_loader.get_all_game_ids_for_user(user)
        user_index = self.node_to_index[user]
        user_scores = (self.path_length_2_weight * (self.matrix[user_index, :] @ self.matrix) + self.path_length_3_weight * (self.matrix[user_index, :] @ self.matrix @ self.matrix)).todense()
        scores = [(game, user_scores[0, self.node_to_index[game]]) for game in self.game_nodes if game not in games_to_filter_out]
        return self.select_scores(scores, N, should_sort)

    def save(self, file_name, overwrite=False):
        assert not os.path.isfile(SAVED_MODELS_PATH + file_name + '.pkl') or overwrite, f'Tried to save to a file that already exists {file_name} without allowing for overwrite.'
        with open(SAVED_MODELS_PATH + file_name + '.pkl', 'wb') as file:
            pickle.dump({
                'path_length_2_weight': self.path_length_2_weight,
                'path_length_3_weight': self.path_length_3_weight,
                'matrix': self.matrix,
                'index_to_node': self.index_to_node,
                'node_to_index': self.node_to_index,
                'game_nodes': self.game_nodes,
            }, file)

    def _load(self, folder_path, file_name):
        with open(folder_path + file_name + '.pkl', 'rb') as file:
            loaded_obj = pickle.load(file)
            self.path_length_2_weight = loaded_obj['path_length_2_weight']
            self.path_length_3_weight = loaded_obj['path_length_3_weight']
            self.matrix = loaded_obj['matrix']
            self.index_to_node = loaded_obj['index_to_node']
            self.node_to_index = loaded_obj['node_to_index']
            self.game_nodes = loaded_obj['game_nodes']
