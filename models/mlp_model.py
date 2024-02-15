from base_model import BaseGameRecommendationModel, SAVED_MODELS_PATH
import numpy as np
from tqdm import tqdm
import random
from matplotlib import pyplot as plt
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.append("../dataset")
sys.path.append("../utils")
from data_loader import NodeType, get_edges_between_types
from utils import linear_transformation, get_numeric_dataframe_columns

class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FeedForwardNetwork, self).__init__()
        if len(hidden_sizes) == 0:
            raise Exception('Need at least 1 hidden layer.')
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x

# MLP
class MLP(BaseGameRecommendationModel):
    def __init__(self, num_epochs = 20, hidden_sizes = [100], learning_rate = 0.01):
        self.num_epochs = num_epochs
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
    
    def name(self):
        return 'mlp'

    def train(self, debug = False):
        self.game_nodes = [node for node, data in self.data_loader.train_network.nodes(data=True) if data['node_type'] == NodeType.GAME]
        self.user_nodes = [node for node, data in self.data_loader.train_network.nodes(data=True) if data['node_type'] == NodeType.USER]
        self.game_to_index = {game: ii for ii, game in enumerate(self.game_nodes)}
        self.user_to_index = {user: ii for ii, user in enumerate(self.user_nodes)}
        
        def normalize_column(column):
            min_val = column.min()
            max_val = column.max()
            normalized_column = linear_transformation(column, min_val, max_val, 0.0, 1.0)
            return normalized_column

        self.game_data_df = get_numeric_dataframe_columns(pd.DataFrame([self.data_loader.train_network.nodes[game_node] for game_node in self.game_nodes]))
        self.game_data_df = self.game_data_df.apply(normalize_column)
        self.known_game_embeddings = self.game_data_df.to_numpy()
        self.known_game_user_embeddings = np.random.rand(len(self.user_nodes), self.known_game_embeddings.shape[1]) / (self.num_user_embedding ** 0.5)
        
        self.user_data_df = get_numeric_dataframe_columns(pd.DataFrame([self.data_loader.train_network.nodes[user_node] for user_node in self.user_nodes]))
        self.user_data_df = self.user_data_df.apply(normalize_column)
        self.known_user_embeddings = self.user_data_df.to_numpy()
        self.known_user_game_embeddings = np.random.rand(len(self.game_nodes), self.known_user_embeddings.shape[1]) / (self.num_game_embedding ** 0.5)

        self.game_embeddings = np.random.rand(len(self.game_nodes), self.num_game_embedding) / (self.num_game_embedding ** 0.5)
        self.user_embeddings = np.random.rand(len(self.user_nodes), self.num_user_embedding) / (self.num_user_embedding ** 0.5)
        print('Total Number of Features:', len(self.user_nodes) * self.num_user_embedding + len(self.game_nodes) * self.num_game_embedding)
        print('Known Game Embeddings: ', self.game_data_df.columns.tolist())
        print('Known User Embeddings: ', self.user_data_df.columns.tolist())

        edges = get_edges_between_types(self.data_loader.train_network, NodeType.USER, NodeType.GAME, data=True)
        # Minimize cosine angle between latent user vectors for friendship
        abs_errors = []
        for epoch in tqdm(range(self.num_epochs)):
            random_edge_index_order = list(range(len(edges)))
            random.shuffle(random_edge_index_order)
            abs_errors.append(0)
            for edge_ii in random_edge_index_order:
                user, game, data = edges[edge_ii]
                user_ii = self.user_to_index[user]
                game_ii = self.game_to_index[game]
                predicted = np.sum(self.user_embeddings[user_ii, :] * self.game_embeddings[game_ii, :]) + np.sum(self.known_game_user_embeddings[user_ii, :] * self.known_game_embeddings[game_ii, :]) + np.sum(self.known_user_embeddings[user_ii, :] * self.known_user_game_embeddings[game_ii, :])
                # print(np.sum(self.user_embeddings[user_ii, :] * self.game_embeddings[game_ii, :]))
                # print(np.sum(self.known_game_user_embeddings[user_ii, :] * self.known_game_embeddings[game_ii, :]))
                # print(np.sum(self.known_user_embeddings[user_ii, :] * self.known_user_game_embeddings[game_ii, :]))
                # print(predicted, data['score'])
                # print(self.known_game_user_embeddings[user_ii, :])
                error = predicted - data['score']
                abs_errors[-1] += abs(error)
                old_user_embeddings = self.user_embeddings[user_ii, :]
                self.user_embeddings[user_ii, :] = self.user_embeddings[user_ii, :] - self.learning_rate * (error * self.game_embeddings[game_ii, :] + self.regularization * self.user_embeddings[user_ii, :])
                self.game_embeddings[game_ii, :] = self.game_embeddings[game_ii, :] - self.learning_rate * (error * old_user_embeddings + self.regularization * self.game_embeddings[game_ii, :])
                self.known_game_user_embeddings[user_ii, :] = self.known_game_user_embeddings[user_ii, :] - self.learning_rate * (error * self.known_game_embeddings[game_ii, :] + self.regularization * self.known_game_user_embeddings[user_ii, :])
                self.known_user_game_embeddings[game_ii, :] = self.known_user_game_embeddings[game_ii, :] - self.learning_rate * (error * self.known_user_embeddings[user_ii, :] + self.regularization * self.known_user_game_embeddings[game_ii, :])
                # print(self.known_game_user_embeddings[user_ii, :])
                # print()
                # input()
            abs_errors[-1] /= len(random_edge_index_order)
        if debug:
            plt.plot(range(self.num_epochs), abs_errors)
            plt.title('Mean Abs Error vs Epoch')

    def score_and_predict_n_games_for_user(self, user, N=None):
        root_node_neighbors = list(self.data_loader.train_network.neighbors(user))
        user_ii = self.user_to_index[user]
        scores = np.sum(self.game_embeddings * self.user_embeddings[user_ii], axis=1) + np.sum(self.known_game_embeddings * self.known_game_user_embeddings[user_ii], axis=1) + np.sum(self.known_user_game_embeddings * self.known_user_embeddings[user_ii], axis=1)
        scores = [(game, {'score': scores[self.game_to_index[game]]}) for game in self.game_nodes if game not in root_node_neighbors]
        scores = sorted(scores, key=lambda x: x[1]['score'], reverse=True)
        if N is not None:
            scores = scores[:N]
        return scores
    
    def predict_for_all_users(self):
        predictions = self.user_embeddings @ self.game_embeddings.T + self.known_game_user_embeddings @ self.known_game_embeddings.T + self.known_user_embeddings @ self.known_user_game_embeddings.T
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
                'known_game_embeddings': self.known_game_embeddings,
                'known_game_user_embeddings': self.known_game_user_embeddings,
                'known_user_embeddings': self.known_user_embeddings,
                'known_user_game_embeddings': self.known_user_game_embeddings,
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
            self.known_game_embeddings = loaded_obj['known_game_embeddings']
            self.known_game_user_embeddings = loaded_obj['known_game_user_embeddings']
            self.known_user_embeddings = loaded_obj['known_user_embeddings']
            self.known_user_game_embeddings = loaded_obj['known_user_game_embeddings']
            self.user_embeddings = loaded_obj['user_embeddings']
            self.game_embeddings = loaded_obj['game_embeddings']
