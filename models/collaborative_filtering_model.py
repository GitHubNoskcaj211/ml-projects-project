from models.base_model import BaseGameRecommendationModel, SAVED_MODELS_PATH
import numpy as np
from tqdm import tqdm
import random
from matplotlib import pyplot as plt
import pickle
import pandas as pd
import math
from pprint import pprint
import time

import os
from dataset.data_loader import NodeType, get_edges_between_types
from utils.utils import linear_transformation, gaussian_transformation, get_numeric_dataframe_columns

MAX_DIFFERENCE = 1e5

# Base Collaborative Filtering
# Based on https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-2-alternating-least-square-als-matrix-4a76c58714a1
class CollaborativeFiltering(BaseGameRecommendationModel):
    def __init__(self, num_epochs = 20, num_user_embedding = 50, num_game_embedding = 50, learning_rate = 0.01, regularization = 0.05, seed=int(time.time())):
        self.num_epochs = num_epochs
        self.num_user_embedding = num_user_embedding
        self.num_game_embedding = num_game_embedding
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.seed = seed
    
    def name(self):
        return 'collaborative_filtering'
    # TODO train with friend cosine scores
    def train(self, train_network, debug = False):
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.game_nodes = [node for node, data in train_network.nodes(data=True) if data['node_type'] == NodeType.GAME]
        self.user_nodes = [node for node, data in train_network.nodes(data=True) if data['node_type'] == NodeType.USER]
        self.game_to_index = {game: ii for ii, game in enumerate(self.game_nodes)}
        self.user_to_index = {user: ii for ii, user in enumerate(self.user_nodes)}
        
        def normalize_column(column):
            normalized_column = gaussian_transformation(column, column.mean(), column.std(), 0.0, min(column.std(), 1.0))
            # normalized_column = linear_transformation(column, column.min(), column.max(), -0.1, 0.1)
            return normalized_column

        self.game_data_df = get_numeric_dataframe_columns(pd.DataFrame([train_network.nodes[game_node] for game_node in self.game_nodes]))
        self.game_data_df = self.game_data_df.apply(normalize_column, axis=0)
        # print(self.game_data_df[['num_reviews']].max(), self.game_data_df[['num_reviews']].min())
        self.known_game_embeddings = self.game_data_df.to_numpy()
        # self.known_game_user_embeddings = np.random.rand(len(self.user_nodes), self.known_game_embeddings.shape[1]) / (self.known_game_embeddings.shape[1] ** 0.5)
        self.known_game_user_embeddings = np.zeros((len(self.user_nodes), self.known_game_embeddings.shape[1]))
        
        self.user_data_df = get_numeric_dataframe_columns(pd.DataFrame([train_network.nodes[user_node] for user_node in self.user_nodes]))
        self.user_data_df = self.user_data_df.apply(normalize_column, axis=0)
        self.known_user_embeddings = self.user_data_df.to_numpy()
        # self.known_user_game_embeddings = np.random.rand(len(self.game_nodes), self.known_user_embeddings.shape[1]) / (self.known_user_embeddings.shape[1] ** 0.5)
        self.known_user_game_embeddings = np.zeros((len(self.game_nodes), self.known_user_embeddings.shape[1]))

        self.game_embeddings = np.random.rand(len(self.game_nodes), self.num_game_embedding) / (self.num_game_embedding ** 0.5)
        self.user_embeddings = np.random.rand(len(self.user_nodes), self.num_user_embedding) / (self.num_user_embedding ** 0.5)
        print('Total Learnable Parameters:', self.game_embeddings.shape[0] * self.game_embeddings.shape[1] + self.user_embeddings.shape[0] * self.user_embeddings.shape[1] + self.known_game_user_embeddings.shape[0] * self.known_game_user_embeddings.shape[1] + self.known_user_game_embeddings.shape[0] * self.known_user_game_embeddings.shape[1])
        print('Known Game Embeddings: ', self.game_data_df.columns.tolist())
        print('Known User Embeddings: ', self.user_data_df.columns.tolist())

        edges = get_edges_between_types(train_network, NodeType.USER, NodeType.GAME, data=True)
        learning_rate_scale = self.user_embeddings.shape[1] + self.game_embeddings.shape[1] + self.known_game_user_embeddings.shape[1] + self.known_user_game_embeddings.shape[1] + self.known_game_embeddings.shape[1] + self.known_user_embeddings.shape[1]
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
                error = predicted - data['score']
                abs_errors[-1] += abs(error)
                old_user_embeddings = self.user_embeddings[user_ii, :]
                old_known_game_user_embeddings = self.known_game_user_embeddings[user_ii, :]
                old_known_user_game_embeddings = self.known_user_game_embeddings[game_ii, :]
                self.user_embeddings[user_ii, :] = self.user_embeddings[user_ii, :] - np.clip(self.learning_rate / learning_rate_scale * (error * self.game_embeddings[game_ii, :] + self.regularization * self.user_embeddings[user_ii, :]), -MAX_DIFFERENCE, MAX_DIFFERENCE)
                self.game_embeddings[game_ii, :] = self.game_embeddings[game_ii, :] - np.clip(self.learning_rate / learning_rate_scale * (error * old_user_embeddings + self.regularization * self.game_embeddings[game_ii, :]), -MAX_DIFFERENCE, MAX_DIFFERENCE)
                self.known_game_user_embeddings[user_ii, :] = self.known_game_user_embeddings[user_ii, :] - np.clip(self.learning_rate / learning_rate_scale * (error * self.known_game_embeddings[game_ii, :] + self.regularization * self.known_game_user_embeddings[user_ii, :]), -MAX_DIFFERENCE, MAX_DIFFERENCE)
                self.known_user_game_embeddings[game_ii, :] = self.known_user_game_embeddings[game_ii, :] - np.clip(self.learning_rate / learning_rate_scale * (error * self.known_user_embeddings[user_ii, :] + self.regularization * self.known_user_game_embeddings[game_ii, :]), -MAX_DIFFERENCE, MAX_DIFFERENCE)
                self.known_game_embeddings[game_ii, :] = self.known_game_embeddings[game_ii, :] - np.clip(self.learning_rate / learning_rate_scale * (error * old_known_game_user_embeddings + self.regularization * self.known_game_embeddings[game_ii, :]), -MAX_DIFFERENCE, MAX_DIFFERENCE)
                self.known_user_embeddings[user_ii, :] = self.known_user_embeddings[user_ii, :] - np.clip(self.learning_rate / learning_rate_scale * (error * old_known_user_game_embeddings + self.regularization * self.known_user_embeddings[user_ii, :]), -MAX_DIFFERENCE, MAX_DIFFERENCE)
            abs_errors[-1] /= len(random_edge_index_order)
        if debug:
            plt.plot(range(self.num_epochs), abs_errors)
            plt.title('Mean Abs Error vs Epoch')
        # TODO at end of training predict for all users so the other methods can just extract scores

    def get_score_between_user_and_game(self, user, game):
        user_ii = self.user_to_index[user]
        game_ii = self.game_to_index[game]
        return np.sum(self.user_embeddings[user_ii, :] * self.game_embeddings[game_ii, :]) + np.sum(self.known_game_user_embeddings[user_ii, :] * self.known_game_embeddings[game_ii, :]) + np.sum(self.known_user_embeddings[user_ii, :] * self.known_user_game_embeddings[game_ii, :])

    def score_and_predict_n_games_for_user(self, user, N=None, should_sort=True):
        games_to_filter_out = self.data_loader.users_games_df[self.data_loader.users_games_df['user_id'] == user]['game_id'].to_list()
        user_ii = self.user_to_index[user]
        scores = np.sum(self.game_embeddings * self.user_embeddings[user_ii], axis=1) + np.sum(self.known_game_embeddings * self.known_game_user_embeddings[user_ii], axis=1) + np.sum(self.known_user_game_embeddings * self.known_user_embeddings[user_ii], axis=1)
        scores = [(game, scores[self.game_to_index[game]]) for game in self.game_nodes if game not in games_to_filter_out]
        return self.select_scores(scores, N, should_sort)

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

    def _load(self, file_path):
        with open(file_path + '.pkl', 'rb') as file:
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
