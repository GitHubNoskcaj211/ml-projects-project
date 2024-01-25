from base_model import BaseGameRecommendationModel, SAVED_MODELS_PATH
import networkx as nx
import pickle
import sys
import os
import random

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../dataset'))
from data_loader import NodeType

# Recommend in order of most to least popular games (based on number of edges).
class GamePopularityModel(BaseGameRecommendationModel):
    def __init__(self):
        pass

    def name(self):
        return 'game_popularity'

    def train(self):
        self.game_nodes = [node for node, data in self.data_loader.train_network.nodes(data=True) if data['node_type'] == NodeType.GAME]
        self.degrees = {node: val for (node, val) in self.data_loader.train_network.degree()}
        score_fn = lambda game: self.degrees[game]
        self.scores = [(game, score_fn(game), None) for game in self.game_nodes]
        self.scores = sorted(self.scores, key=lambda x: x[1], reverse=True)


    def score_and_predict_n_games_for_user(self, user, N=None):
        root_node_neighbors = list(self.data_loader.train_network.neighbors(user))
        scores_for_user = [(game, score, embeddings) for game, score, embeddings in self.scores if game not in root_node_neighbors]
        if N is not None:
            scores_for_user = scores_for_user[:N]
        return scores_for_user

    def save(self, file_name, overwrite=False):
        raise NotImplementedError('Did not implement saving on popularity model.')

    def load(self, file_name):
        raise NotImplementedError('Did not implement loading on popularity model.')
