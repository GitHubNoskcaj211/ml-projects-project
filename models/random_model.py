from base_model import BaseGameRecommendationModel, SAVED_MODELS_PATH
import networkx as nx
import pickle
import sys
import os
import random

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../dataset'))
from data_loader import NodeType

class RandomModel(BaseGameRecommendationModel):
    def __init__(self, seed = 0):
        self.seed = 0

    def name(self):
        return 'random'

    def train(self):
        random.seed(self.seed)
        self.game_nodes = [node for node, data in self.data_loader.train_network.nodes(data=True) if data['node_type'] == NodeType.GAME]

    def score_and_predict_n_games_for_user(self, user, N=None):
        root_node_neighbors = list(self.data_loader.train_network.neighbors(user))
        score_fn = lambda user, game: random.random()
        scores = [(game, {'score': score_fn(user, game)}) for game in self.game_nodes if game not in root_node_neighbors]
        scores = sorted(scores, key=lambda x: x[1]['score'], reverse=True)
        if N is not None:
            scores = scores[:N]
        return scores

    def save(self, file_name, overwrite=False):
        raise NotImplementedError('Did not implement saving on random model.')

    def load(self, file_name):
        raise NotImplementedError('Did not implement loading on random model.')
