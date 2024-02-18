from abc import ABC, abstractmethod
import numpy as np
from quickselect import floyd_rivest
import multiprocessing

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../dataset'))
from data_loader import NodeType

SAVED_MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models/')
SAVED_NN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_nns/')

class BaseGameRecommendationModel(ABC):
    @abstractmethod
    def name(self):
        pass

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    def select_and_sort_scores(self, scores, N = None):
        if N is not None and N < len(scores):
            nth_largest_score = floyd_rivest.nth_largest(scores, N - 1, key=lambda score: score[1]['score'])[1]['score']
            scores = [score for score in scores if score[1]['score'] >= nth_largest_score]
        scores = sorted(scores, key=lambda x: x[1]['score'], reverse=True)
        if N is not None:
            scores = scores[:N]
        return scores
    
    # def select_and_sort_scores(self, scores, N = None):
    #     scores = sorted(scores, key=lambda x: x[1]['score'], reverse=True)
    #     if N is not None:
    #         scores = scores[:N]
    #     return scores

    # Train the model given the data loader.
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def get_embeddings_between_user_and_game(self, user, game):
        pass

    # Input: 
    # Output: list of nodes 
    def recommend_n_games_for_user(self, user, N=None):
        scores = self.score_and_predict_n_games_for_user(user, N)
        return [game for game, embeddings in scores]

    # def predict_for_all_users(self, N):
    #     all_predictions_and_scores_per_user = {}
    #     pool = multiprocessing.Pool(processes=16, maxtasksperchild=1)
    #     for node, data in self.data_loader.test_network.nodes(data=True):
    #         if data['node_type'] != NodeType.USER:
    #             continue
    #         all_predictions_and_scores_per_user[node] = pool.apply_async(self.score_and_predict_n_games_for_user, args=(node, N))
    #     pool.close()
    #     pool.join()
    #     for node, async_result in all_predictions_and_scores_per_user.items():
    #         all_predictions_and_scores_per_user[node] = async_result.get()
    #     return all_predictions_and_scores_per_user

    def predict_for_all_users(self, N):
        all_predictions_and_scores_per_user = {}
        for node, data in self.data_loader.test_network.nodes(data=True):
            if data['node_type'] != NodeType.USER:
                continue
            all_predictions_and_scores_per_user[node] = self.score_and_predict_n_games_for_user(node, N=N)
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