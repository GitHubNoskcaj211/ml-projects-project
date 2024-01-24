from base_model import BaseGameRecommendationModel, SAVED_MODELS_PATH
import networkx as nx
import pickle
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../dataset'))
from data_loader import NodeType

class CommonNeighbors(BaseGameRecommendationModel):
    def __init__(self, path_length_2_weight = 1, path_length_3_weight = 1):
        self.path_length_2_weight = path_length_2_weight
        self.path_length_3_weight = path_length_3_weight

    def name(self):
        return 'common_neighbors'

    def train(self):
        self.matrix = nx.adjacency_matrix(self.data_loader.train_network)
        self.length_2_paths = (self.matrix @ self.matrix).todense()
        self.length_3_paths = (self.matrix @ self.matrix @ self.matrix).todense()
        self.index_to_node = list(self.data_loader.train_network.nodes())
        self.node_to_index = {node: ii for ii, node in enumerate(self.data_loader.train_network.nodes())}
        self.game_nodes = [node for node, data in self.data_loader.train_network.nodes(data=True) if data['node_type'] == NodeType.GAME]

    def score_and_predict_n_games_for_user(self, user, N=None):
        root_node_neighbors = list(self.data_loader.train_network.neighbors(user))
        score_fn = lambda user, game: self.path_length_2_weight * self.length_2_paths[self.node_to_index[user]][self.node_to_index[game]] + self.path_length_3_weight * self.length_3_paths[self.node_to_index[user]][self.node_to_index[game]]
        scores = [(game, score_fn(user, game), None) for game in self.game_nodes if game not in root_node_neighbors]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        if N is not None:
            scores = scores[:N]
        return scores

    def save(self, file_name, overwrite=False):
        assert not os.path.isfile(SAVED_MODELS_PATH + file_name) or overwrite, f'Tried to save to a file that already exists {file_name} without allowing for overwrite.'
        with open(SAVED_MODELS_PATH + file_name, 'wb') as file:
            pickle.dump({
                'path_length_2_weight': self.path_length_2_weight,
                'path_length_3_weight': self.path_length_3_weight,
                'matrix': self.matrix,
                'length_2_paths': self.length_2_paths,
                'length_3_paths': self.length_3_paths,
                'index_to_node': self.index_to_node,
                'node_to_index': self.node_to_index,
                'game_nodes': self.game_nodes,
            }, file)

    def load(self, file_name):
        with open(SAVED_MODELS_PATH + file_name, 'rb') as file:
            loaded_obj = pickle.load(file)
            self.path_length_2_weight = loaded_obj['path_length_2_weight']
            self.path_length_3_weight = loaded_obj['path_length_3_weight']
            self.matrix = loaded_obj['matrix']
            self.length_2_paths = loaded_obj['length_2_paths']
            self.length_3_paths = loaded_obj['length_3_paths']
            self.index_to_node = loaded_obj['index_to_node']
            self.node_to_index = loaded_obj['node_to_index']
            self.game_nodes = loaded_obj['game_nodes']
