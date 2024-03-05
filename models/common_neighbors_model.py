from models.base_model import BaseGameRecommendationModel, SAVED_MODELS_PATH
import networkx as nx
import pickle
import sys
import os

from dataset.data_loader import NodeType

# TODO Add unit tests to ensure these are equivalent.
class CommonNeighborsModelStoragePredictEfficient(BaseGameRecommendationModel):
    def __init__(self, path_length_2_weight = 1, path_length_3_weight = 1):
        self.path_length_2_weight = path_length_2_weight
        self.path_length_3_weight = path_length_3_weight

    def name(self):
        return 'common_neighbors'

    def train(self, train_network):
        self.matrix = nx.adjacency_matrix(train_network)
        self.index_to_node = list(train_network.nodes())
        self.node_to_index = {node: ii for ii, node in enumerate(train_network.nodes())}
        self.game_nodes = [node for node, data in train_network.nodes(data=True) if data['node_type'] == NodeType.GAME]
        
        self.length_2_paths = (self.matrix @ self.matrix).todense()
        self.length_3_paths = (self.matrix @ self.matrix @ self.matrix).todense()
        self.scores = self.path_length_2_weight * self.length_2_paths + self.path_length_3_weight * self.length_3_paths

    def get_score_between_user_and_game(self, user, game):
        return self.scores[self.node_to_index[user], self.node_to_index[game]]

    def score_and_predict_n_games_for_user(self, user, N=None, should_sort=True):
        games_to_filter_out = self.data_loader.users_games_df[self.data_loader.users_games_df['user_id'] == user]['game_id'].to_list()
        scores = [(game, self.scores[self.node_to_index[user], self.node_to_index[game]]) for game in self.game_nodes if game not in games_to_filter_out]
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

    def load(self, file_name):
        with open(SAVED_MODELS_PATH + file_name + '.pkl', 'rb') as file:
            loaded_obj = pickle.load(file)
            self.path_length_2_weight = loaded_obj['path_length_2_weight']
            self.path_length_3_weight = loaded_obj['path_length_3_weight']
            self.matrix = loaded_obj['matrix']
            self.index_to_node = loaded_obj['index_to_node']
            self.node_to_index = loaded_obj['node_to_index']
            self.game_nodes = loaded_obj['game_nodes']

            self.length_2_paths = (self.matrix @ self.matrix).todense()
            self.length_3_paths = (self.matrix @ self.matrix @ self.matrix).todense()
            self.scores = self.path_length_2_weight * self.length_2_paths + self.path_length_3_weight * self.length_3_paths

class CommonNeighborsModelLoadPredictEfficient(BaseGameRecommendationModel):
    def __init__(self, path_length_2_weight = 1, path_length_3_weight = 1):
        self.path_length_2_weight = path_length_2_weight
        self.path_length_3_weight = path_length_3_weight

    def name(self):
        return 'common_neighbors'

    def train(self, train_network):
        matrix = nx.adjacency_matrix(train_network)
        self.index_to_node = list(train_network.nodes())
        self.node_to_index = {node: ii for ii, node in enumerate(train_network.nodes())}
        self.game_nodes = [node for node, data in train_network.nodes(data=True) if data['node_type'] == NodeType.GAME]
        
        length_2_paths = (matrix @ matrix).todense()
        length_3_paths = (matrix @ matrix @ matrix).todense()
        self.scores = self.path_length_2_weight * length_2_paths + self.path_length_3_weight * length_3_paths

    def get_score_between_user_and_game(self, user, game):
        return self.scores[self.node_to_index[user], self.node_to_index[game]]

    def score_and_predict_n_games_for_user(self, user, N=None, should_sort=True):
        games_to_filter_out = self.data_loader.users_games_df[self.data_loader.users_games_df['user_id'] == user]['game_id'].to_list()
        scores = [(game, self.scores[self.node_to_index[user], self.node_to_index[game]]) for game in self.game_nodes if game not in games_to_filter_out]
        return self.select_scores(scores, N, should_sort)

    def save(self, file_name, overwrite=False):
        assert not os.path.isfile(SAVED_MODELS_PATH + file_name + '.pkl') or overwrite, f'Tried to save to a file that already exists {file_name} without allowing for overwrite.'
        with open(SAVED_MODELS_PATH + file_name + '.pkl', 'wb') as file:
            pickle.dump({
                'path_length_2_weight': self.path_length_2_weight,
                'path_length_3_weight': self.path_length_3_weight,
                'index_to_node': self.index_to_node,
                'node_to_index': self.node_to_index,
                'game_nodes': self.game_nodes,
                'scores': self.scores,
            }, file)

    def load(self, file_name):
        with open(SAVED_MODELS_PATH + file_name + '.pkl', 'rb') as file:
            loaded_obj = pickle.load(file)
            self.path_length_2_weight = loaded_obj['path_length_2_weight']
            self.path_length_3_weight = loaded_obj['path_length_3_weight']
            self.index_to_node = loaded_obj['index_to_node']
            self.node_to_index = loaded_obj['node_to_index']
            self.game_nodes = loaded_obj['game_nodes']
            self.scores = loaded_obj['scores']

class CommonNeighborsModelStorageMemoryEfficient(BaseGameRecommendationModel):
    def __init__(self, path_length_2_weight = 1, path_length_3_weight = 1):
        self.path_length_2_weight = path_length_2_weight
        self.path_length_3_weight = path_length_3_weight

    def name(self):
        return 'common_neighbors'

    def train(self, train_network):
        self.matrix = nx.adjacency_matrix(train_network)
        self.index_to_node = list(train_network.nodes())
        self.node_to_index = {node: ii for ii, node in enumerate(train_network.nodes())}
        self.game_nodes = [node for node, data in train_network.nodes(data=True) if data['node_type'] == NodeType.GAME]

    def get_score_between_user_and_game(self, user, game):
        user_index = self.node_to_index[user]
        game_index = self.node_to_index[game]
        return self.path_length_2_weight * (self.matrix[[user_index], :] @ self.matrix[:, [game_index]])[0, 0] + self.path_length_3_weight * (self.matrix[[user_index], :] @ self.matrix @ self.matrix[:, [game_index]])[0, 0]

    def score_and_predict_n_games_for_user(self, user, N=None, should_sort=True):
        games_to_filter_out = self.data_loader.users_games_df[self.data_loader.users_games_df['user_id'] == user]['game_id'].to_list()
        user_index = self.node_to_index[user]
        user_scores = (self.path_length_2_weight * (self.matrix[[user_index], :] @ self.matrix) + self.path_length_3_weight * (self.matrix[[user_index], :] @ self.matrix @ self.matrix)).todense()[0]
        scores = [(game, user_scores[self.node_to_index[game]]) for game in self.game_nodes if game not in games_to_filter_out]
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

    def _load(self, file_path):
        with open(file_path + '.pkl', 'rb') as file:
            loaded_obj = pickle.load(file)
            self.path_length_2_weight = loaded_obj['path_length_2_weight']
            self.path_length_3_weight = loaded_obj['path_length_3_weight']
            self.matrix = loaded_obj['matrix']
            self.index_to_node = loaded_obj['index_to_node']
            self.node_to_index = loaded_obj['node_to_index']
            self.game_nodes = loaded_obj['game_nodes']