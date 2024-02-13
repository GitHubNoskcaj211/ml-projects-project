from abc import ABC, abstractmethod
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../dataset'))
from data_loader import NodeType

SAVED_MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models/')

class BaseGameRecommendationModel(ABC):
    @abstractmethod
    def name(self):
        pass

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    # Train the model given the data loader.
    @abstractmethod
    def train(self):
        pass

    # Input: 
    # Output: list of nodes 
    def recommend_n_games_for_user(self, user, N=None):
        scores = self.score_n_games_for_user(user, N)
        return [game for game, embeddings in scores]

    def predict_for_all_users(self):
        all_predictions_and_scores_per_user = {}
        for node, data in self.data_loader.test_network.nodes(data=True):
            if data['node_type'] != NodeType.USER:
                continue
            all_predictions_and_scores_per_user[node] = self.score_and_predict_n_games_for_user(node, N=None)
        return all_predictions_and_scores_per_user

    # Output: List of top N scores (sorted) for new game recommendations for a user. Formatted as [(game_id, embedding_predictions)] where embedding_predictions is a dictionary
    @abstractmethod
    def score_and_predict_n_games_for_user(self, user, N):
        pass

    @abstractmethod
    def save(self, file_name, overwrite=False):
        pass

    @abstractmethod
    def load(self, file_name):
        pass