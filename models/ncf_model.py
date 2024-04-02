from models.base_model import BaseGameRecommendationModel, SAVED_MODELS_PATH
from models.ncf_singlenode import NCF
import pickle
import time
import torch
import pandas as pd

import sys
import os
sys.path.append("../dataset")
sys.path.append("../utils")
from utils.utils import linear_transformation, gaussian_transformation, get_numeric_dataframe_columns

class NCFModel(BaseGameRecommendationModel):
    def __init__(self, num_epochs = 20, embedding_size = 100, batch_percent = 0.1, learning_rate = 0.01, weight_decay=1e-5, mlp_hidden_layer_sizes = [16], seed=int(time.time()), model_type='ncf', fine_tune_num_epochs=1, fine_tune_learning_rate=1e-1, fine_tune_weight_decay=1e-5):
        super().__init__()
        self.num_epochs = num_epochs
        self.embedding_size = embedding_size
        self.batch_percent = batch_percent
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.mlp_hidden_layer_sizes = mlp_hidden_layer_sizes
        self.seed = seed
        self.model_type = model_type
        self.fine_tune_num_epochs = fine_tune_num_epochs
        self.fine_tune_learning_rate = fine_tune_learning_rate
        self.fine_tune_weight_decay = fine_tune_weight_decay
    
    def name(self):
        return f'neural_collborative_filtering_{self.model_type}'
    
    def train(self, debug = False):
        # TODO train on downloaded interactions
        assert self.data_loader.cache_local_dataset, 'Method requires full load.'
        self.game_nodes = self.data_loader.get_game_node_ids()
        self.user_nodes = self.data_loader.get_user_node_ids()
        self.game_to_index = {game: ii for ii, game in enumerate(self.game_nodes)}
        self.user_to_index = {user: ii for ii, user in enumerate(self.user_nodes)}
        train_users_games_df = self.data_loader.users_games_df[self.data_loader.users_games_df['data_split'] == 'train']
        
        def normalize_column(column):
            normalized_column = gaussian_transformation(column, column.mean(), column.std(), 0.0, min(column.std(), 1.0))
            # normalized_column = linear_transformation(column, column.min(), column.max(), -0.1, 0.1)
            return normalized_column

        # TODO use existing data
        game_data_df = get_numeric_dataframe_columns(self.data_loader.games_df, columns_to_remove=['id'])
        game_data_df = game_data_df.apply(normalize_column, axis=0)
        # known_game_embeddings = game_data_df.to_numpy()
        
        user_data_df = get_numeric_dataframe_columns(self.data_loader.users_df, columns_to_remove=['id'])
        user_data_df = user_data_df.apply(normalize_column, axis=0)
        # known_user_embeddings = user_data_df.to_numpy()
        
        print('Known Game Embeddings: ', game_data_df.columns.tolist())
        print('Known User Embeddings: ', user_data_df.columns.tolist())

        user_indices = torch.tensor(train_users_games_df['user_id'].apply(lambda id: self.user_to_index[id]).values)
        game_indices = torch.tensor(train_users_games_df['game_id'].apply(lambda id: self.game_to_index[id]).values)

        self.num_users = len(self.user_nodes)
        self.num_games = len(self.game_nodes)
        self.ncf = NCF(self.num_users, self.num_games, self.model_type, self.embedding_size, self.mlp_hidden_layer_sizes, self.num_epochs, self.batch_percent, self.learning_rate, self.weight_decay, self.seed)

        user_game_scores_tensor = torch.tensor(train_users_games_df['score'].values)
        user_game_scores_tensor = user_game_scores_tensor.type(torch.FloatTensor)
        user_game_scores_tensor = torch.reshape(user_game_scores_tensor, (-1, 1))
        self.ncf.train(user_indices, game_indices, user_game_scores_tensor, debug)

    def test_loss(self):
        test_users_games_df = self.data_loader.users_games_df[self.data_loader.users_games_df['data_split'] == 'test']
        user_indices = torch.tensor(test_users_games_df['user_id'].apply(lambda id: self.user_to_index[id]).values)
        game_indices = torch.tensor(test_users_games_df['game_id'].apply(lambda id: self.game_to_index[id]).values)
        user_game_scores_tensor = torch.tensor(test_users_games_df['score'].values)
        user_game_scores_tensor = user_game_scores_tensor.type(torch.FloatTensor)
        user_game_scores_tensor = torch.reshape(user_game_scores_tensor, (-1, 1))
        return self.ncf.test_loss(user_indices, game_indices, user_game_scores_tensor)

    def _fine_tune(self, user_id, new_user_games_df, new_interactions_df, all_user_games_df, all_interactions_df):
        if new_user_games_df.empty and new_interactions_df.empty:
            return
        if not user_id in self.user_to_index:
            self.ncf.add_new_user()
            self.user_to_index[user_id] = len(self.user_nodes)
            self.user_nodes.append(user_id)
        # TODO use all new ones and sample some already trained ones.
        user_indices = pd.concat([new_user_games_df['user_id'].apply(lambda id: self.user_to_index[id]), new_interactions_df['user_id'].apply(lambda id: self.user_to_index[id])])
        user_indices = torch.tensor(user_indices.values)
        game_indices = pd.concat([new_user_games_df['game_id'].apply(lambda id: self.game_to_index[id]), new_interactions_df['game_id'].apply(lambda id: self.game_to_index[id])])
        game_indices = torch.tensor(game_indices.values)
        scores = pd.concat([new_user_games_df['score'], new_interactions_df['score']])
        scores_tensor = torch.tensor(scores.values)
        scores_tensor = scores_tensor.type(torch.FloatTensor)
        scores_tensor = torch.reshape(scores_tensor, (-1, 1))
        # TODO parameterize these later.
        self.fine_tune_num_epochs = 40
        self.fine_tune_weight_decay = 1e-1#1e-3
        self.fine_tune_learning_rate = 1e-1
        self.ncf.fine_tune(self.user_to_index[user_id], user_indices, game_indices, scores_tensor, self.fine_tune_num_epochs, self.fine_tune_learning_rate, self.fine_tune_weight_decay, debug=False)
            

    def get_score_between_user_and_game(self, user, game):
        user_ii = self.user_to_index[user]
        game_ii = self.game_to_index[game]
        output = self.ncf.predict(torch.tensor([user_ii]), torch.tensor([game_ii]))
        return output[0][0]

    def score_and_predict_n_games_for_user(self, user, N=None, should_sort=True):
        games_to_filter_out = self.data_loader.get_all_game_ids_for_user(user)
        user_ii = self.user_to_index[user]
        game_indices = [self.game_to_index[game] for game in self.game_nodes if game not in games_to_filter_out]
        output = self.ncf.predict(torch.tensor([user_ii] * len(game_indices)), torch.tensor(game_indices))
        scores = [(self.game_nodes[game_ii], game_output[0]) for game_ii, game_output in zip(game_indices, output)]
        return self.select_scores(scores, N, should_sort)

    def new_seed(self, seed=None):
        self.ncf.new_seed(seed)

    def save(self, file_name, overwrite=False):
        assert not os.path.isfile(SAVED_MODELS_PATH + file_name + '.pkl') or overwrite, f'Tried to save to a file that already exists {file_name} without allowing for overwrite.'
        with open(SAVED_MODELS_PATH + file_name + '.pkl', 'wb') as file:
            pickle.dump({
                'num_epochs': self.num_epochs,
                'embedding_size': self.embedding_size,
                'batch_percent': self.batch_percent,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'mlp_hidden_layer_sizes': self.mlp_hidden_layer_sizes,
                'seed': self.seed,
                'model_type': self.model_type,
                'num_users': self.num_users,
                'num_games': self.num_games,
                'game_nodes': self.game_nodes,
                'user_nodes': self.user_nodes,
                'game_to_index': self.game_to_index,
                'user_to_index': self.user_to_index,
                'fine_tune_num_epochs': self.fine_tune_num_epochs,
                'fine_tune_learning_rate': self.fine_tune_learning_rate,
                'fine_tune_weight_decay': self.fine_tune_weight_decay,
            }, file)
        self.ncf.save(file_name, overwrite=overwrite)

    def _load(self, folder_path, file_name):
        with open(folder_path + file_name + '.pkl', 'rb') as file:
            loaded_obj = pickle.load(file)
            self.num_epochs = loaded_obj['num_epochs']
            self.embedding_size = loaded_obj['embedding_size']
            self.batch_percent = loaded_obj['batch_percent']
            self.learning_rate = loaded_obj['learning_rate']
            self.weight_decay = loaded_obj['weight_decay']
            self.mlp_hidden_layer_sizes = loaded_obj['mlp_hidden_layer_sizes']
            self.seed = loaded_obj['seed']
            self.model_type = loaded_obj['model_type']
            self.num_users = loaded_obj['num_users']
            self.num_games = loaded_obj['num_games']
            self.game_nodes = loaded_obj['game_nodes']
            self.user_nodes = loaded_obj['user_nodes']
            self.game_to_index = loaded_obj['game_to_index']
            self.user_to_index = loaded_obj['user_to_index']
            self.fine_tune_num_epochs = loaded_obj['fine_tune_num_epochs']
            self.fine_tune_learning_rate = loaded_obj['fine_tune_learning_rate']
            self.fine_tune_weight_decay = loaded_obj['fine_tune_weight_decay']
        self.ncf = NCF(self.num_users, self.num_games, self.model_type, self.embedding_size, self.mlp_hidden_layer_sizes, self.num_epochs, self.batch_percent, self.learning_rate, self.weight_decay, self.seed)
        self.ncf.load(folder_path, file_name)