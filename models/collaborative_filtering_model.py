from base_model import BaseGameRecommendationModel
import numpy as np
from tqdm import tqdm
import random

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

    def train(self, data_loader, debug = False):
        self.user_embeddings = np.random.rand(<num_users>, self.num_user_embedding) / (self.num_user_embedding ** 0.5)
        self.game_embeddings = np.random.rand(<num_games>, self.num_game_embedding) / (self.num_game_embedding ** 0.5)
        print('Total Number of Features:', <num_users> * self.num_user_embedding + <num_games> * self.num_game_embedding)
        
        edges = list(self.network.edges())
        target_for_all_edges = 1
        abs_errors = []
        for epoch in tqdm(range(self.num_epochs)):
            random_edge_index_order = list(range(len(edges)))
            random.shuffle(random_edge_index_order)
            abs_errors.append(0)
            for edge_ii in random_edge_index_order:
                # TODO Filter to only user - game edges.
                edge = edges[edge_ii]
                user_node = edge[0] if 'user' in edge[0] else edge[1]
                game_node = edge[0] if 'game' in edge[0] else edge[1]
                user_ii = int(user_node.split('_')[1])
                game_ii = int(game_node.split('_')[1])
                predicted_score = np.sum(self.user_embeddings[user_ii, :] * self.game_embeddings[game_ii, :])
                error = predicted_score - target_for_all_edges
                abs_errors[-1] += abs(error)
                old_user_embeddings = self.user_embeddings[user_ii, :]
                self.user_embeddings[user_ii, :] = self.user_embeddings[user_ii, :] - self.learning_rate * (error * self.game_embeddings[game_ii, :] + self.regularization * self.user_embeddings[user_ii, :])
                self.game_embeddings[game_ii, :] = self.game_embeddings[game_ii, :] - self.learning_rate * (error * old_user_embeddings + self.regularization * self.game_embeddings[game_ii, :])
        print('training errors', abs_errors)

    def recommend_n_games_for_user(self, user, N):
        node_neighbors = list(self.network.neighbors(user))
        user_ii = int(user.split('_')[1])
        scores = np.sum(self.game_embeddings * self.user_embeddings[user_ii], axis=1)
        scores = [(f'game_{ii}', score) for ii, score in enumerate(scores) if f'game_{ii}' not in node_neighbors]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        if N is not None:
            scores = scores[:N]
        return [recommendation for recommendation, score in scores]

    def save(self, file_name, overwrite=False):
        pass

    def load(self, file_name):
        pass   
