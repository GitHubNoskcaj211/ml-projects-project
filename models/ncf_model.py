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
from collections import defaultdict
import itertools

import sys
import os
sys.path.append("../dataset")
sys.path.append("../utils")
from data_loader import NodeType, get_edges_between_types
from utils import linear_transformation, gaussian_transformation, get_numeric_dataframe_columns

class NCFModel(BaseGameRecommendationModel):
    def __init__(self, num_epochs = 20, embedding_size = 100, batch_percent = 0.1, learning_rate = 0.01, mlp_hidden_layer_sizes = [16], seed=int(time.time()), model_type='ncf'):
        self.num_epochs = num_epochs
        self.embedding_size = embedding_size
        self.batch_percent = batch_percent
        self.learning_rate = learning_rate
        self.mlp_hidden_layer_sizes = mlp_hidden_layer_sizes
        self.seed = seed
        self.model_type = model_type

    def normalize_score(self, score):
        return linear_transformation(score, self.min_score, self.max_score, 0, 1)

    def unnormalize_score(self, score):
        return linear_transformation(score, 0, 1, self.min_score, self.max_score)
    
    def name(self):
        return f'neural_collborative_filtering_{self.model_type}'
    
    def train(self, debug = False):
        self.game_nodes = [node for node, data in self.data_loader.train_network.nodes(data=True) if data['node_type'] == NodeType.GAME]
        self.user_nodes = [node for node, data in self.data_loader.train_network.nodes(data=True) if data['node_type'] == NodeType.USER]
        self.game_to_index = {game: ii for ii, game in enumerate(self.game_nodes)}
        self.user_to_index = {user: ii for ii, user in enumerate(self.user_nodes)}
        
        def normalize_column(column):
            normalized_column = gaussian_transformation(column, column.mean(), column.std(), 0.0, min(column.std(), 1.0))
            # normalized_column = linear_transformation(column, column.min(), column.max(), -0.1, 0.1)
            return normalized_column

        game_data_df = get_numeric_dataframe_columns(pd.DataFrame([self.data_loader.train_network.nodes[game_node] for game_node in self.game_nodes]))
        game_data_df = game_data_df.apply(normalize_column, axis=0)
        # known_game_embeddings = game_data_df.to_numpy()
        
        user_data_df = get_numeric_dataframe_columns(pd.DataFrame([self.data_loader.train_network.nodes[user_node] for user_node in self.user_nodes]))
        user_data_df = user_data_df.apply(normalize_column, axis=0)
        # known_user_embeddings = user_data_df.to_numpy()
        
        print('Known Game Embeddings: ', game_data_df.columns.tolist())
        print('Known User Embeddings: ', user_data_df.columns.tolist())

        user_game_edges = list(get_edges_between_types(self.data_loader.train_network, NodeType.USER, NodeType.GAME, data=True))
        user_game_scores = np.array([data['score'] for user, game, data in user_game_edges])
        user_game_scores = user_game_scores.reshape((-1, 1))
        user_indices = torch.tensor([self.user_to_index[user] for user, game, data in user_game_edges])
        game_indices = torch.tensor([self.game_to_index[game] for user, game, data in user_game_edges])
        self.min_score = np.min(user_game_scores)
        self.max_score = np.max(user_game_scores)
        if self.min_score == 1 and self.max_score == 1:
            self.min_score = 0
            self.max_score = 1
        user_game_scores = self.normalize_score(user_game_scores)

        self.num_users = len(self.user_nodes)
        self.num_games = len(self.game_nodes)
        self.ncf = NCF(self.num_users, self.num_games, self.model_type, self.embedding_size, self.mlp_hidden_layer_sizes, self.num_epochs, self.batch_percent, self.learning_rate, self.seed)

        user_game_scores_tensor = torch.tensor(user_game_scores)
        user_game_scores_tensor = user_game_scores_tensor.type(torch.FloatTensor)
        self.ncf.train(user_indices, game_indices, user_game_scores_tensor, debug)

    def get_score_between_user_and_game(self, user, game):
        user_ii = self.user_to_index[user]
        game_ii = self.game_to_index[game]
        output = self.ncf.predict(torch.tensor([user_ii]), torch.tensor([game_ii]))
        return self.unnormalize_score(output[0][0])

    def score_and_predict_n_games_for_user(self, user, N=None, should_sort=True):
        root_node_neighbors = list(self.data_loader.train_network.neighbors(user))
        user_ii = self.user_to_index[user]
        game_indices = [self.game_to_index[game] for game in self.game_nodes if game not in root_node_neighbors]
        output = self.ncf.predict(torch.tensor([user_ii] * len(game_indices)), torch.tensor(game_indices))
        scores = [(self.game_nodes[game_ii], self.unnormalize_score(game_output[0])) for game_ii, game_output in zip(game_indices, output)]
        return self.select_scores(scores, N, should_sort)
    
    # TODO This runs out of memory figure out why
    # def predict_for_all_users(self, N):
    #     all_predictions_and_scores_per_user = defaultdict(list)
    #     user_indices = [self.user_to_index[user] for user in self.user_nodes]
    #     game_indices = [self.game_to_index[game] for game in self.game_nodes]
    #     all_user_game_combinations = list(itertools.product(user_indices, game_indices))
    #     output = self.ncf.predict(torch.tensor([user for user, game in all_user_game_combinations]), torch.tensor([game for user, game in all_user_game_combinations]), is_list=True)
    #     for (user_ii, game_ii), user_game_output in zip(all_user_game_combinations, output):
    #         if self.game_nodes[game_ii] in self.data_loader.train_network[self.user_nodes[user_ii]]:
    #             continue
    #         all_predictions_and_scores_per_user[self.user_nodes[user_ii]].append((self.game_nodes[game_ii], {self.output_index_to_embedding_name[ii]: value for ii, value in enumerate(user_game_output)}))
    #     for user, scores in tqdm(all_predictions_and_scores_per_user.items(), desc='User Predictions'):
    #         all_predictions_and_scores_per_user[user] = self.select_scores(scores, N)
    #     return all_predictions_and_scores_per_user

    def save(self, file_name, overwrite=False):
        return
        assert not os.path.isfile(SAVED_MODELS_PATH + file_name + '.pkl') or overwrite, f'Tried to save to a file that already exists {file_name} without allowing for overwrite.'
        with open(SAVED_MODELS_PATH + file_name + '.pkl', 'wb') as file:
            pickle.dump({
                'game_nodes': self.game_nodes,
                'user_nodes': self.user_nodes,
                'game_to_index': self.game_to_index,
                'user_to_index': self.user_to_index,
                'known_game_embeddings': known_game_embeddings,
                'known_game_user_embeddings': self.known_game_user_embeddings,
                'known_user_embeddings': known_user_embeddings,
                'known_user_game_embeddings': self.known_user_game_embeddings,
                'user_embeddings': self.user_embeddings,
                'game_embeddings': self.game_embeddings,
                'seed': self.seed,
            }, file)

    def load(self, file_name):
        return
        with open(SAVED_MODELS_PATH + file_name + '.pkl', 'rb') as file:
            loaded_obj = pickle.load(file)
            self.game_nodes = loaded_obj['game_nodes']
            self.user_nodes = loaded_obj['user_nodes']
            self.game_to_index = loaded_obj['game_to_index']
            self.user_to_index = loaded_obj['user_to_index']
            known_game_embeddings = loaded_obj['known_game_embeddings']
            self.known_game_user_embeddings = loaded_obj['known_game_user_embeddings']
            known_user_embeddings = loaded_obj['known_user_embeddings']
            self.known_user_game_embeddings = loaded_obj['known_user_game_embeddings']
            self.user_embeddings = loaded_obj['user_embeddings']
            self.game_embeddings = loaded_obj['game_embeddings']
            self.seed = loaded_obj['seed']
