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
from utils.utils import gaussian_transformation, get_numeric_dataframe_columns

if "K_SERVICE" not in os.environ:
    import datetime
    from torch.utils.tensorboard import SummaryWriter

TENSORBOARD_RUN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tensorboard_runs/')

class NCFModel(BaseGameRecommendationModel):
    def __init__(self, num_epochs = 20, embedding_size = 100, batch_percent = 0.1, learning_rate = 0.01, weight_decay=1e-5, mlp_hidden_layer_sizes = [16], seed=int(time.time()), model_type='ncf', fine_tune_num_epochs=1, fine_tune_learning_rate=1e-1, fine_tune_weight_decay=1e-5, save_file_name=None, nn_save_name=None):
        super().__init__()
        self.save_file_name = save_file_name
        self.nn_save_name = nn_save_name
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
    
    def train(self, debug = False, user_node_ids=None):
        # TODO train on downloaded interactions
        assert self.data_loader.cache_local_dataset, 'Method requires full load.'
        self.game_nodes = self.data_loader.get_game_node_ids()
        self.user_nodes = user_node_ids if user_node_ids is not None else self.data_loader.get_user_node_ids()
        self.game_to_index = {game: ii for ii, game in enumerate(self.game_nodes)}
        self.user_to_index = {user: ii for ii, user in enumerate(self.user_nodes)}
        train_users_games_df = self.data_loader.users_games_df[self.data_loader.users_games_df['data_split'] == 'train']
        
        def normalize_column(column):
            normalized_column = gaussian_transformation(column, column.mean(), column.std(), 0.0, min(column.std(), 1.0))
            normalized_column = normalized_column.clip(-5, 5)
            return normalized_column

        assert all((id1 == id2 for id1, id2 in zip(self.game_nodes, self.data_loader.games_df['id']))), 'Need the dataframe ids to have the same order as get_game_node_ids for embeddings to assign properly.'
        self.known_game_embeddings_df = get_numeric_dataframe_columns(self.data_loader.games_df, columns_to_remove=['id'])
        self.known_game_embeddings_df = self.known_game_embeddings_df.apply(normalize_column, axis=0)
        # for column in self.known_game_embeddings_df.columns:
        #     plt.figure()
        #     plt.hist(self.known_game_embeddings_df[column], bins=100, color='skyblue', edgecolor='black')
        #     plt.title(f'Distribution of {column}')
        #     plt.xlabel('Value')
        #     plt.ylabel('Frequency')
        #     plt.yscale('log')
        #     plt.show()
        print('Known Game Embeddings: ', self.known_game_embeddings_df.columns.tolist())

        user_indices = torch.tensor(train_users_games_df['user_id'].apply(lambda id: self.user_to_index[id]).values)
        game_indices = torch.tensor(train_users_games_df['game_id'].apply(lambda id: self.game_to_index[id]).values)

        self.num_users = len(self.user_nodes)
        self.num_games = len(self.game_nodes)
        self.ncf = NCF(self.num_users, self.num_games, self.model_type, self.embedding_size, self.known_game_embeddings_df, self.mlp_hidden_layer_sizes, self.num_epochs, self.batch_percent, self.learning_rate, self.weight_decay, self.seed)

        user_game_scores_tensor = torch.tensor(train_users_games_df['score'].values)
        user_game_scores_tensor = user_game_scores_tensor.type(torch.FloatTensor)
        user_game_scores_tensor = torch.reshape(user_game_scores_tensor, (-1, 1))

        test_users_games_df = self.data_loader.users_games_df[(self.data_loader.users_games_df['data_split'] == 'test') & (self.data_loader.users_games_df['user_id'].isin(self.user_nodes))]
        test_user_indices = torch.tensor(test_users_games_df['user_id'].apply(lambda id: self.user_to_index[id]).values)
        test_game_indices = torch.tensor(test_users_games_df['game_id'].apply(lambda id: self.game_to_index[id]).values)
        test_user_game_scores_tensor = torch.tensor(test_users_games_df['score'].values)
        test_user_game_scores_tensor = test_user_game_scores_tensor.type(torch.FloatTensor)
        test_user_game_scores_tensor = torch.reshape(test_user_game_scores_tensor, (-1, 1))

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(os.path.join(TENSORBOARD_RUN_PATH, f"{self.save_file_name}_{current_time}"))
        self.ncf.train(user_indices, game_indices, user_game_scores_tensor, test_user_indices, test_game_indices, test_user_game_scores_tensor, f'{self.save_file_name}_best', f'{self.save_file_name}_last', debug, writer)

    def _fine_tune(self, user_id, new_user_games_df, new_interactions_df, all_user_games_df, all_interactions_df, debug = False):
        if not user_id in self.user_to_index:
            self.ncf.add_new_user()
            self.user_to_index[user_id] = len(self.user_nodes)
            self.user_nodes.append(user_id)
        if new_user_games_df.empty and new_interactions_df.empty:
            return
        user_indices = pd.concat([all_user_games_df['user_id'].apply(lambda id: self.user_to_index[id]), all_interactions_df['user_id'].apply(lambda id: self.user_to_index[id])])
        user_indices = torch.tensor(user_indices.values)
        game_indices = pd.concat([all_user_games_df['game_id'].apply(lambda id: self.game_to_index[id]), all_interactions_df['game_id'].apply(lambda id: self.game_to_index[id])])
        game_indices = torch.tensor(game_indices.values)
        scores = pd.concat([all_user_games_df['score'], all_interactions_df['score']])
        scores_tensor = torch.tensor(scores.values)
        scores_tensor = scores_tensor.type(torch.FloatTensor)
        scores_tensor = torch.reshape(scores_tensor, (-1, 1))
        # TODO parameterize these later.
        self.fine_tune_num_epochs = 100
        self.fine_tune_weight_decay = 1e-6#1e-3
        self.fine_tune_learning_rate = 1e-2

        writer = None
        test_user_indices = None
        test_game_indices = None
        test_scores_tensor = None
        if debug:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            writer = SummaryWriter(os.path.join(TENSORBOARD_RUN_PATH, f"{self.save_file_name}_{user_id}_{current_time}"))
            assert self.data_loader.cache_local_dataset, 'Need to cache local dataset to debug.'
            users_games_df_for_user = self.data_loader.users_games_df_grouped_by_user.get_group(user_id)
            test_user_games_df = users_games_df_for_user[users_games_df_for_user['data_split'] == 'test']
            test_user_indices = pd.concat([test_user_games_df['user_id'].apply(lambda id: self.user_to_index[id])])
            test_user_indices = torch.tensor(test_user_indices.values)
            test_game_indices = pd.concat([test_user_games_df['game_id'].apply(lambda id: self.game_to_index[id])])
            test_game_indices = torch.tensor(test_game_indices.values)
            test_scores = pd.concat([test_user_games_df['score']])
            test_scores_tensor = torch.tensor(test_scores.values)
            test_scores_tensor = test_scores_tensor.type(torch.FloatTensor)
            test_scores_tensor = torch.reshape(test_scores_tensor, (-1, 1))
        
        self.ncf.fine_tune(self.user_to_index[user_id], user_indices, game_indices, scores_tensor, self.fine_tune_num_epochs, self.fine_tune_learning_rate, self.fine_tune_weight_decay, debug=debug, writer=writer, test_user_indices=test_user_indices, test_game_indices=test_game_indices, test_scores=test_scores_tensor)
            

    def get_score_between_user_and_game(self, user, game):
        user_ii = self.user_to_index[user]
        game_ii = self.game_to_index[game]
        output = self.ncf.predict(torch.tensor([user_ii]), torch.tensor([game_ii]))
        return output[0]
    
    def get_scores_between_users_and_games(self, users, games):
        assert len(users) == len(games), 'Inconsistent list lengths.'
        users_ii = [self.user_to_index[user] for user in users]
        games_ii = [self.game_to_index[game] for game in games]
        output = self.ncf.predict(torch.tensor(users_ii), torch.tensor(games_ii))
        return output

    def score_and_predict_n_games_for_user(self, user, N=None, should_sort=True, games_to_include=[]):
        games_to_filter_out = self.data_loader.get_all_game_ids_for_user(user)
        user_ii = self.user_to_index[user]
        game_indices = [self.game_to_index[game] for game in self.game_nodes if game not in games_to_filter_out]
        output = self.ncf.predict(torch.tensor([user_ii] * len(game_indices)), torch.tensor(game_indices))
        scores = [(self.game_nodes[game_ii], game_score) for game_ii, game_score in zip(game_indices, output)]
        return self.select_scores(scores, N, should_sort, games_to_include=games_to_include)

    def new_seed(self, seed=None):
        self.ncf.new_seed(seed)

    def save(self, file_name, overwrite=False):
        assert not os.path.isfile(SAVED_MODELS_PATH + file_name + '.pkl') or overwrite, f'Tried to save to a file that already exists {file_name} without allowing for overwrite.'
        assert file_name == self.save_file_name, 'Model name in saved parameters must match the requested model.'
        with open(SAVED_MODELS_PATH + file_name + '.pkl', 'wb') as file:
            pickle.dump({
                'save_file_name': self.save_file_name,
                'nn_save_name': self.nn_save_name,
                'num_epochs': self.num_epochs,
                'embedding_size': self.embedding_size,
                'known_game_embeddings_df': self.known_game_embeddings_df,
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

    def _load(self, folder_path, file_name):
        with open(folder_path + file_name + '.pkl', 'rb') as file:
            loaded_obj = pickle.load(file)
            self.save_file_name = loaded_obj['save_file_name']
            self.nn_save_name = loaded_obj['nn_save_name']
            self.num_epochs = loaded_obj['num_epochs']
            self.embedding_size = loaded_obj['embedding_size']
            self.known_game_embeddings_df = loaded_obj['known_game_embeddings_df']
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
        # assert file_name == self.save_file_name, 'Model name in saved parameters must match the requested model.'
        self.ncf = NCF(self.num_users, self.num_games, self.model_type, self.embedding_size, self.known_game_embeddings_df, self.mlp_hidden_layer_sizes, self.num_epochs, self.batch_percent, self.learning_rate, self.weight_decay, self.seed)
        self.ncf.load(folder_path, f'{file_name}_{self.nn_save_name}')