from base_model import BaseGameRecommendationModel, SAVED_MODELS_PATH
import numpy as np
from tqdm import tqdm
import random
from matplotlib import pyplot as plt
import pickle

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../dataset'))
from data_loader import NodeType, get_edges_between_types, filter_numeric_data

# Base Collaborative Filtering
# Based on https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-2-alternating-least-square-als-matrix-4a76c58714a1
class CollaborativeFiltering(BaseGameRecommendationModel):
    def __init__(self, num_epochs = 20, num_user_embedding = 50, num_game_embedding = 50, learning_rate = 0.01, regularization = 0.05):
        self.num_epochs = num_epochs
        self.num_user_embedding = num_user_embedding
        self.num_game_embedding = num_game_embedding
        self.learning_rate = learning_rate
        self.regularization = regularization
    
    def name(self):
        return 'collaborative_filtering'

    def train(self, debug = False):
        self.game_nodes = [node for node, data in self.data_loader.train_network.nodes(data=True) if data['node_type'] == NodeType.GAME]
        self.user_nodes = [node for node, data in self.data_loader.train_network.nodes(data=True) if data['node_type'] == NodeType.USER]
        self.game_to_index = {game: ii for ii, game in enumerate(self.game_nodes)}
        self.user_to_index = {user: ii for ii, user in enumerate(self.user_nodes)}
        self.user_embeddings = np.random.rand(len(self.user_nodes), self.num_user_embedding) / (self.num_user_embedding ** 0.5)
        self.game_embeddings = np.random.rand(len(self.game_nodes), self.num_game_embedding) / (self.num_game_embedding ** 0.5)
        print('Total Number of Features:', len(self.user_nodes) * self.num_user_embedding + len(self.game_nodes) * self.num_game_embedding)
        
        edges = get_edges_between_types(self.data_loader.train_network, NodeType.USER, NodeType.GAME, data=True)

        abs_errors = []
        for epoch in tqdm(range(self.num_epochs)):
            random_edge_index_order = list(range(len(edges)))
            random.shuffle(random_edge_index_order)
            abs_errors.append(0)
            for edge_ii in random_edge_index_order:
                edge, data = edges[edge_ii]
                user_ii = self.user_to_index[edge[0]]
                game_ii = self.game_to_index[edge[1]]
                predicted = np.sum(self.user_embeddings[user_ii, :] * self.game_embeddings[game_ii, :])
                error = predicted - data['score']
                abs_errors[-1] += abs(error)
                old_user_embeddings = self.user_embeddings[user_ii, :]
                self.user_embeddings[user_ii, :] = self.user_embeddings[user_ii, :] - self.learning_rate * (error * self.game_embeddings[game_ii, :] + self.regularization * self.user_embeddings[user_ii, :])
                self.game_embeddings[game_ii, :] = self.game_embeddings[game_ii, :] - self.learning_rate * (error * old_user_embeddings + self.regularization * self.game_embeddings[game_ii, :])
        
        if debug:
            plt.plot(range(self.num_epochs), abs_errors)
            plt.title('Absolute Errors vs Epochs')

    def score_and_predict_n_games_for_user(self, user, N=None):
        root_node_neighbors = list(self.data_loader.train_network.neighbors(user))
        user_ii = self.user_to_index[user]
        scores = np.sum(self.game_embeddings * self.user_embeddings[user_ii], axis=1)
        scores = [(game, {'score': scores[self.game_to_index[game]]}) for game in self.game_nodes if game not in root_node_neighbors]
        scores = sorted(scores, key=lambda x: x[1]['score'], reverse=True)
        if N is not None:
            scores = scores[:N]
        return scores
    
    def predict_for_all_users(self):
        predictions = self.user_embeddings @ self.game_embeddings.T
        all_predictions_and_scores_per_user = {}
        for node, data in self.data_loader.test_network.nodes(data=True):
            if data['node_type'] != NodeType.USER:
                continue
            root_node_neighbors = list(self.data_loader.train_network.neighbors(node))
            user_ii = self.user_to_index[node]
            scores = predictions[user_ii]
            scores = [(game, {'score': scores[self.game_to_index[game]]}) for game in self.game_nodes if game not in root_node_neighbors]
            scores = sorted(scores, key=lambda x: x[1]['score'], reverse=True)
            all_predictions_and_scores_per_user[node] = scores
        return all_predictions_and_scores_per_user

    def save(self, file_name, overwrite=False):
        assert not os.path.isfile(SAVED_MODELS_PATH + file_name) or overwrite, f'Tried to save to a file that already exists {file_name} without allowing for overwrite.'
        with open(SAVED_MODELS_PATH + file_name, 'wb') as file:
            pickle.dump({
                'game_nodes': self.game_nodes,
                'user_nodes': self.user_nodes,
                'game_to_index': self.game_to_index,
                'user_to_index': self.user_to_index,
                'user_embeddings': self.user_embeddings,
                'game_embeddings': self.game_embeddings,
            }, file)

    def load(self, file_name):
        with open(SAVED_MODELS_PATH + file_name, 'rb') as file:
            loaded_obj = pickle.load(file)
            self.game_nodes = loaded_obj['game_nodes']
            self.user_nodes = loaded_obj['user_nodes']
            self.game_to_index = loaded_obj['game_to_index']
            self.user_to_index = loaded_obj['user_to_index']
            self.user_embeddings = loaded_obj['user_embeddings']
            self.game_embeddings = loaded_obj['game_embeddings']


# # Base Collaborative Filtering
# # Based on https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-2-alternating-least-square-als-matrix-4a76c58714a1
# class CollaborativeFiltering(BaseGameRecommendationModel):
#     def __init__(self, num_epochs = 20, num_user_embedding = 50, num_game_embedding = 50, learning_rate = 0.01, regularization = 0.05):
#         self.num_epochs = num_epochs
#         self.num_user_embedding = num_user_embedding
#         self.num_game_embedding = num_game_embedding
#         self.learning_rate = learning_rate
#         self.regularization = regularization
    
#     def name(self):
#         return 'collaborative_filtering'

#     def train(self, data_loader, debug = False):
#         self.game_nodes = [node for node, data in self.data_loader.train_network.nodes(data=True) if data['node_type'] == NodeType.GAME]
#         self.user_nodes = [node for node, data in self.data_loader.train_network.nodes(data=True) if data['node_type'] == NodeType.USER]
#         self.game_to_index = {game: ii for ii, game in enumerate(self.game_nodes)}
#         self.user_to_index = {user: ii for ii, user in enumerate(self.user_nodes)}
#         self.user_embeddings = np.random.rand(len(self.user_nodes), self.num_user_embedding) / (self.num_user_embedding ** 0.5)
#         self.game_embeddings = np.random.rand(len(self.game_nodes), self.num_game_embedding) / (self.num_game_embedding ** 0.5)
#         print('Total Number of Features:', len(self.user_nodes) * self.num_user_embedding + len(self.game_nodes) * self.num_game_embedding)
        
#         edges = get_edges_between_types(data_loader.train_network, NodeType.USER, NodeType.GAME, data=True)
#         edges = [(edge, filter_numeric_data(data)) for edge, data in edges]

#         edge_data_key_to_index = {}
#         cc = 0
#         for edge, data in edges:
#             for key, value in data:
#                 if key not in edge_data_key_to_index:
#                     edge_data_key_to_index[key] = cc
#                     cc += 1

#         encoded_network = np.full((len(self.user_nodes), len(self.game_nodes), len(edge_data_key_to_index)), -1)
#         for edge, data in edges:
#             encoded_edge = np.full((len(edge_data_key_to_index)), -1)
#             for key, value in data.items():
#                 encoded_edge[edge_data_key_to_index[key]] = value
#             encoded_network[edge[0], edge[1], :] = encoded_edge

#         abs_errors = []
#         for epoch in tqdm(range(self.num_epochs)):
#             random_edge_index_order = list(range(len(edges)))
#             random.shuffle(random_edge_index_order)
#             abs_errors.append(0)
#             for edge_ii in random_edge_index_order:
#                 edge = edges[edge_ii]
#                 user_ii = self.user_to_index[edge[0]]
#                 game_ii = self.game_to_index[edge[1]]
#                 predicted = np.sum(self.user_embeddings[user_ii, :] * self.game_embeddings[game_ii, :])
#                 error = predicted - encoded_network[user_ii, game_ii, :]
#                 abs_errors[-1] += abs(error)
#                 old_user_embeddings = self.user_embeddings[user_ii, :]
#                 self.user_embeddings[user_ii, :] = self.user_embeddings[user_ii, :] - self.learning_rate * (error * self.game_embeddings[game_ii, :] + self.regularization * self.user_embeddings[user_ii, :])
#                 self.game_embeddings[game_ii, :] = self.game_embeddings[game_ii, :] - self.learning_rate * (error * old_user_embeddings + self.regularization * self.game_embeddings[game_ii, :])
#         print('training errors', abs_errors)

#     def recommend_n_games_for_user(self, user, N):
#         node_neighbors = list(self.network.neighbors(user))
#         user_ii = int(user.split('_')[1])
#         scores = np.sum(self.game_embeddings * self.user_embeddings[user_ii], axis=1)
#         scores = [(f'game_{ii}', score) for ii, score in enumerate(scores) if f'game_{ii}' not in node_neighbors]
#         scores = sorted(scores, key=lambda x: x[1], reverse=True)
#         if N is not None:
#             scores = scores[:N]
#         return [recommendation for recommendation, score in scores]

#     def save(self, file_name, overwrite=False):
#         pass

#     def load(self, file_name):
#         pass
