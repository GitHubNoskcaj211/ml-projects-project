from base_model import BaseGameRecommendationModel, SAVED_MODELS_PATH
from ncf_singlenode import NCF
import numpy as np
from tqdm import tqdm
import random
from matplotlib import pyplot as plt
import pickle
import pandas as pd
import math
from pprint import pprint
import time
import torch

import sys
import os
sys.path.append("../dataset")
sys.path.append("../utils")
from data_loader import NodeType, get_edges_between_types
from utils import linear_transformation, gaussian_transformation, get_numeric_dataframe_columns

class NCFModel(BaseGameRecommendationModel):
    def __init__(self, num_epochs = 20, embedding_size = 100, batch_percent = 0.1, learning_rate = 0.01, mlp_hidden_layer_sizes = [16], seed=int(time.time()), model_type='neumf'):
        self.num_epochs = num_epochs
        self.embedding_size = embedding_size
        self.batch_percent = batch_percent
        self.learning_rate = learning_rate
        self.mlp_hidden_layer_sizes = mlp_hidden_layer_sizes
        self.seed = seed
        self.model_type = model_type
    
    def name(self):
        return f'neural_collborative_filtering_{self.model_type}'
    
    def train(self, debug = False):
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.game_nodes = [node for node, data in self.data_loader.train_network.nodes(data=True) if data['node_type'] == NodeType.GAME]
        self.user_nodes = [node for node, data in self.data_loader.train_network.nodes(data=True) if data['node_type'] == NodeType.USER]
        self.game_to_index = {game: ii for ii, game in enumerate(self.game_nodes)}
        self.user_to_index = {user: ii for ii, user in enumerate(self.user_nodes)}
        
        def normalize_column(column):
            normalized_column = gaussian_transformation(column, column.mean(), column.std(), 0.0, min(column.std(), 1.0))
            # normalized_column = linear_transformation(column, column.min(), column.max(), -0.1, 0.1)
            return normalized_column

        self.game_data_df = get_numeric_dataframe_columns(pd.DataFrame([self.data_loader.train_network.nodes[game_node] for game_node in self.game_nodes]))
        self.game_data_df = self.game_data_df.apply(normalize_column, axis=0)
        self.known_game_embeddings = self.game_data_df.to_numpy()
        
        self.user_data_df = get_numeric_dataframe_columns(pd.DataFrame([self.data_loader.train_network.nodes[user_node] for user_node in self.user_nodes]))
        self.user_data_df = self.user_data_df.apply(normalize_column, axis=0)
        self.known_user_embeddings = self.user_data_df.to_numpy()
        
        print('Known Game Embeddings: ', self.game_data_df.columns.tolist())
        print('Known User Embeddings: ', self.user_data_df.columns.tolist())

        edges = list(get_edges_between_types(self.data_loader.train_network, NodeType.USER, NodeType.GAME, data=True))
        self.edge_df = get_numeric_dataframe_columns(pd.DataFrame([data for user, game, data in edges]))
        self.output_index_to_embedding_name = {ii: column for ii, column in self.edge_df.columns.tolist()}
        self.edge_data = self.edge_data.to_numpy()

        self.num_users = len(self.user_nodes)
        self.num_games = len(self.game_nodes)
        self.output_size = self.edge_data.shape[1]
        self.ncf = NCF(self.num_users, self.num_games, self.model_type, self.embedding_size, self.mlp_hidden_layer_sizes, self.num_epochs, self.batch_percent, self.learning_rate, self.output_size, self.seed)

        user_indices = [self.user_to_index[user] for user, game, data in self.edges]
        game_indices = [self.game_to_index[game] for user, game, data in self.edges]
        data = torch.tensor(self.edge_data)
        self.ncf.train(user_indices, game_indices, data, debug)

    def get_embeddings_between_user_and_game(self, user, game):
        user_ii = self.user_to_index[user]
        game_ii = self.game_to_index[game]
        return {'score': np.sum(self.user_embeddings[user_ii, :] * self.game_embeddings[game_ii, :]) + np.sum(self.known_game_user_embeddings[user_ii, :] * self.known_game_embeddings[game_ii, :]) + np.sum(self.known_user_embeddings[user_ii, :] * self.known_user_game_embeddings[game_ii, :])}

    def score_and_predict_n_games_for_user(self, user, N=None):
        root_node_neighbors = list(self.data_loader.train_network.neighbors(user))
        user_ii = self.user_to_index[user]
        scores = np.sum(self.game_embeddings * self.user_embeddings[user_ii], axis=1) + np.sum(self.known_game_embeddings * self.known_game_user_embeddings[user_ii], axis=1) + np.sum(self.known_user_game_embeddings * self.known_user_embeddings[user_ii], axis=1)
        scores = [(game, {'score': scores[self.game_to_index[game]]}) for game in self.game_nodes if game not in root_node_neighbors]
        return self.select_and_sort_scores(scores, N)
    
    def predict_for_all_users(self, N):
        predictions = self.user_embeddings @ self.game_embeddings.T + self.known_game_user_embeddings @ self.known_game_embeddings.T + self.known_user_embeddings @ self.known_user_game_embeddings.T
        all_predictions_and_scores_per_user = {}
        for node, data in self.data_loader.test_network.nodes(data=True):
            if data['node_type'] != NodeType.USER:
                continue
            root_node_neighbors = list(self.data_loader.train_network.neighbors(node))
            user_ii = self.user_to_index[node]
            scores = predictions[user_ii]
            scores = [(game, {'score': scores[self.game_to_index[game]]}) for game in self.game_nodes if game not in root_node_neighbors]
            all_predictions_and_scores_per_user[node] = self.select_and_sort_scores(scores, N)
        return all_predictions_and_scores_per_user

    def save(self, file_name, overwrite=False):
        assert not os.path.isfile(SAVED_MODELS_PATH + file_name + '.pkl') or overwrite, f'Tried to save to a file that already exists {file_name} without allowing for overwrite.'
        with open(SAVED_MODELS_PATH + file_name + '.pkl', 'wb') as file:
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
                'seed': self.seed,
            }, file)

    def load(self, file_name):
        with open(SAVED_MODELS_PATH + file_name + '.pkl', 'rb') as file:
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
            self.seed = loaded_obj['seed']
