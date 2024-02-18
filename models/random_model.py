from base_model import BaseGameRecommendationModel, SAVED_MODELS_PATH
import networkx as nx
import pickle
import sys
import os
import random
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../dataset'))
from data_loader import NodeType

class RandomModel(BaseGameRecommendationModel):
    def __init__(self, seed = 0):
        self.seed = seed

    def name(self):
        return 'random'

    def train(self):
        np.random.seed(self.seed)
        self.user_nodes = [node for node, data in self.data_loader.train_network.nodes(data=True) if data['node_type'] == NodeType.USER]
        self.game_nodes = [node for node, data in self.data_loader.train_network.nodes(data=True) if data['node_type'] == NodeType.GAME]
        self.user_to_user_index = {user: ii for ii, user in enumerate(self.user_nodes)}
        self.game_to_game_index = {game: ii for ii, game in enumerate(self.game_nodes)}
        self.random_scores = np.random.random((len(self.user_nodes), len(self.game_nodes)))

    def get_embeddings_between_user_and_game(self, user, game):
        return {'score': self.random_scores[self.user_to_user_index[user], self.game_to_game_index[game]]}

    def score_and_predict_n_games_for_user(self, user, N=None):
        root_node_neighbors = list(self.data_loader.train_network.neighbors(user))
        user_index = self.user_to_user_index[user]
        scores = [(game, {'score': self.random_scores[user_index, self.game_to_game_index[game]]}) for game in self.game_nodes if game not in root_node_neighbors]
        return self.select_and_sort_scores(scores, N)

    def save(self, file_name, overwrite=False):
        raise NotImplementedError('Did not implement saving on random model.')

    def load(self, file_name):
        raise NotImplementedError('Did not implement loading on random model.')
