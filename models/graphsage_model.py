from models.base_model import BaseGameRecommendationModel, SAVED_MODELS_PATH, SAVED_NN_PATH
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from utils.utils import gaussian_transformation, get_numeric_dataframe_columns
import pickle
import pandas as pd

import os
if "K_SERVICE" not in os.environ:
    import datetime
    from torch.utils.tensorboard import SummaryWriter
TENSORBOARD_RUN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tensorboard_runs/')

# from torch_geometric.datasets import MovieLens
# dataset = MovieLens('~/temp1/', model_name='all-MiniLM-L6-v2')
# print(dataset[0])

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.conv3 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)

        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['user'][row], z_dict['game'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class HeterogeneousGraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels, data_metadata, aggr, num_epochs, batch_percent, learning_rate, weight_decay, seed):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data_metadata, aggr=aggr)
        self.decoder = EdgeDecoder(hidden_channels)
        self.num_epochs = num_epochs
        self.batch_percent = batch_percent
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.new_seed(seed)

        self.loss_fn = nn.L1Loss()

    def new_seed(self, seed=None):
        if seed is None:
            torch.seed()
        else:
            torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed

    def forward(self, x_dict, edge_index_dict, edge_score_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_score_index)

    def get_batch_for_users(self, train_data, user_indices=None):
        fill_value = True if user_indices is None else False
        users_to_include = torch.full((len(train_data['user']['x']),), fill_value)
        if user_indices is not None:
            for user_index in user_indices:
                users_to_include[user_index] = True
        num_users_to_include = int(torch.sum(users_to_include.int()))
        loader = NeighborLoader(
            train_data,
            num_neighbors={key: [30, 30, 30] for key in train_data.edge_types},
            batch_size=num_users_to_include,
            input_nodes=('user', users_to_include),
        )
        batch = next(iter(loader))
        data = HeteroData()
        data['user'].x = train_data['user'].x
        data['game'].x = train_data['game'].x
        data['user', 'plays', 'game'].edge_index = train_data['user', 'plays', 'game'].edge_index[:, batch['user', 'plays', 'game'].e_id]
        data['game', 'rev_plays', 'user'].edge_index = train_data['game', 'rev_plays', 'user'].edge_index[:, batch['game', 'rev_plays', 'user'].e_id]
        return data

    def train(self, train_data, test_edge_index, test_labels, optimal_model_save_name, last_model_save_name, debug=False, writer=None):
        super().train(True)
        # Initializes the network.
        self.forward(train_data.x_dict, {key: value[:, 0:1] for key, value in train_data.edge_index_dict.items()},
                    train_data['user', 'game'].edge_index[:, 0:1])

        for p in self.parameters():
            p.requires_grad_(True)
        
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)
        print('Total Learnable Parameters:', sum(p.numel() for p in self.parameters() if p.requires_grad))
        batch_size = int(len(train_data['user']['x']) * self.batch_percent) + 1
        loader = NeighborLoader(
            train_data,
            num_neighbors={key: [30, 30, 30] for key in train_data.edge_types},
            batch_size=batch_size,
            input_nodes=('user', torch.full((len(train_data['user']['x']),), True)),
        )

        train_loss = []
        lowest_test_loss = None
        for epoch_count in tqdm(range(self.num_epochs), desc='Training'):
            epoch_loss = []
            for batch in loader:
                pred = self.forward(batch.x_dict, batch.edge_index_dict, batch['user', 'game'].edge_index)
                # TODO We perform a new round of negative sampling for every training epoch?
                target = batch['user', 'game'].edge_label
                loss = self.loss_fn(pred, target)
                loss.backward()
                optimizer.step()

                epoch_loss.append(loss.item())
            average_epoch_loss = sum((value for value in epoch_loss)) / len(epoch_loss)
            train_loss.append(average_epoch_loss)
            test_loss = self.test(train_data, test_edge_index, test_labels)
            if writer is not None:
                writer.add_scalar('Loss/train', average_epoch_loss, epoch_count)
                writer.add_scalar('Loss/test', test_loss, epoch_count)
            if optimal_model_save_name is not None:
                # TODO Use val loss instead of test loss.
                if lowest_test_loss is None or test_loss < lowest_test_loss:
                    lowest_test_loss = test_loss
                    self._save(optimal_model_save_name, overwrite=True)
            self._save(last_model_save_name, overwrite=True)
    
    @torch.no_grad()
    def test(self, train_data, test_edge_index, test_labels):
        super().train(False)
        batch = self.get_batch_for_users(train_data, user_indices=None)
        pred = self.forward(batch.x_dict, batch.edge_index_dict, test_edge_index)
        loss = self.loss_fn(pred, test_labels)
        super().train(True)
        return loss.item()

    @torch.no_grad()
    def predict(self, train_data, user_indices, edge_score_index):
        super().train(False)
        batch = self.get_batch_for_users(train_data, user_indices)
        output = self.forward(batch.x_dict, batch.edge_index_dict, edge_score_index)
        return output.detach().flatten().tolist()

    def _save(self, file_name, overwrite=False):
        assert not os.path.isfile(SAVED_MODELS_PATH + SAVED_NN_PATH + file_name + '.pth') or overwrite, f'Tried to save to a file that already exists {file_name} without allowing for overwrite.'
        torch.save(self.state_dict(), os.path.join(SAVED_MODELS_PATH, SAVED_NN_PATH, file_name + '.pth'))

    def load(self, folder_path, file_name):
        self.load_state_dict(torch.load(os.path.join(folder_path, SAVED_NN_PATH, file_name + '.pth')))


class GraphSAGE(BaseGameRecommendationModel):
    def __init__(self, hidden_channels=50, aggr='mean', save_file_name=None, nn_save_name=None, num_epochs=50, batch_percent=0.1, learning_rate=5e-3, weight_decay=1e-5, seed=None):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.aggr = aggr
        self.save_file_name = save_file_name
        self.nn_save_name = nn_save_name
        self.num_epochs = num_epochs
        self.batch_percent = batch_percent
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed

    def name(self):
        return "graphsage"
    
    def new_seed(self, seed=None):
        self.model.new_seed(seed)

    def train(self, debug=False, user_node_ids=None):
        assert self.data_loader.cache_local_dataset, 'Method requires full load.'
        self.game_nodes = self.data_loader.get_game_node_ids()
        self.user_nodes = user_node_ids if user_node_ids is not None else self.data_loader.get_user_node_ids()
        self.game_to_index = {game: ii for ii, game in enumerate(self.game_nodes)}
        self.user_to_index = {user: ii for ii, user in enumerate(self.user_nodes)}
        train_users_games_df = self.data_loader.users_games_df[self.data_loader.users_games_df['data_split'] == 'train']
        test_users_games_df = self.data_loader.users_games_df[self.data_loader.users_games_df['data_split'] == 'test']

        self.data = HeteroData()
        self.data['user'].x = torch.full((len(self.user_nodes), 1), 1, dtype=torch.float32)

        def normalize_column(column):
            normalized_column = gaussian_transformation(column, column.mean(), column.std(), 0.0, min(column.std(), 1.0))
            normalized_column = normalized_column.clip(-5, 5)
            return normalized_column

        assert all((id1 == id2 for id1, id2 in zip(self.game_nodes, self.data_loader.games_df['id']))), 'Need the dataframe ids to have the same order as get_game_node_ids for embeddings to assign properly.'
        self.known_game_embeddings_df = get_numeric_dataframe_columns(self.data_loader.games_df, columns_to_remove=['id'])
        self.known_game_embeddings_df = self.known_game_embeddings_df.apply(normalize_column, axis=0)
        known_game_embeddings_tensor = torch.tensor(self.known_game_embeddings_df.values, dtype=torch.float32)
        # game_tensor = torch.cat((known_game_embeddings_tensor, torch.eye(known_game_embeddings_tensor.shape[0])), dim=1)
        game_tensor = known_game_embeddings_tensor
        self.data['game'].x = game_tensor

        user_indices = torch.tensor(train_users_games_df['user_id'].apply(lambda id: self.user_to_index[id]).values).reshape((1, -1))
        game_indices = torch.tensor(train_users_games_df['game_id'].apply(lambda id: self.game_to_index[id]).values).reshape((1, -1))
        self.data['user', 'plays', 'game'].edge_index = torch.cat((user_indices, game_indices), dim=0)
        user_game_scores_tensor = torch.tensor(train_users_games_df['score'].values)
        user_game_scores_tensor = user_game_scores_tensor.type(torch.FloatTensor)
        self.data['user', 'plays', 'game'].edge_label = user_game_scores_tensor

        self.data = T.ToUndirected()(self.data)
        del self.data['game', 'rev_plays', 'user'].edge_label

        self.data_metadata = self.data.metadata()
        self.model = HeterogeneousGraphSAGE(self.hidden_channels, self.data_metadata, self.aggr, self.num_epochs, self.batch_percent, self.learning_rate, self.weight_decay, self.seed)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(os.path.join(TENSORBOARD_RUN_PATH, f"{self.save_file_name}_{current_time}"))

        test_user_indices = torch.tensor(test_users_games_df['user_id'].apply(lambda id: self.user_to_index[id]).values).reshape((1, -1))
        test_game_indices = torch.tensor(test_users_games_df['game_id'].apply(lambda id: self.game_to_index[id]).values).reshape((1, -1))
        test_edge_index = torch.cat((test_user_indices, test_game_indices), dim=0)
        test_labels = torch.tensor(test_users_games_df['score'].values)
        test_labels = test_labels.type(torch.FloatTensor)

        self.model.train(self.data, test_edge_index, test_labels, f'{self.save_file_name}_best', f'{self.save_file_name}_last', debug=debug, writer=writer)

    def get_score_between_user_and_game(self, user_id, game_id, data):
        return None

    def get_scores_between_users_and_games(self, user_id, game_id):
        return None

    def score_and_predict_n_games_for_user(self, user_id, N=None, should_sort=True, games_to_include=[]):
        games_to_filter_out = self.data_loader.get_all_game_ids_for_user(user_id)
        user_index = self.user_to_index[user_id]

        user_indices = torch.full((1, len(self.game_nodes)), user_index, dtype=torch.long)
        game_indices = torch.tensor([self.game_to_index[game_id] for game_id in self.game_nodes]).reshape((1, -1))
        edge_score_index = torch.cat((user_indices, game_indices), dim=0)
        scores = self.model.predict(self.data, [user_index], edge_score_index)
        scores = [(game_id, score) for game_id, score in zip(self.game_nodes, scores)]

        return self.select_scores(scores, N=N, should_sort=should_sort, games_to_filter_out=games_to_filter_out, games_to_include=games_to_include)

    def save(self, file_name, overwrite=False):
        assert not os.path.isfile(SAVED_MODELS_PATH + file_name + '.pkl') or overwrite, f'Tried to save to a file that already exists {file_name} without allowing for overwrite.'
        assert file_name == self.save_file_name, 'Model name in saved parameters must match the requested model.'
        with open(SAVED_MODELS_PATH + file_name + '.pkl', 'wb') as file:
            pickle.dump({
                'save_file_name': self.save_file_name,
                'nn_save_name': self.nn_save_name,
                'hidden_channels': self.hidden_channels,
                'data_metadata': self.data_metadata,
                'aggr': self.aggr,
                'num_epochs': self.num_epochs,
                'batch_percent': self.batch_percent,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'seed': self.seed,
                'game_nodes': self.game_nodes,
                'user_nodes': self.user_nodes,
                'game_to_index': self.game_to_index,
                'user_to_index': self.user_to_index,
                'data': self.data,
            }, file)

    def _load(self, folder_path, file_name):
        with open(folder_path + file_name + '.pkl', 'rb') as file:
            loaded_obj = pickle.load(file)
            self.save_file_name = loaded_obj['save_file_name']
            self.nn_save_name = loaded_obj['nn_save_name']
            self.hidden_channels = loaded_obj['hidden_channels']
            self.data_metadata = loaded_obj['data_metadata']
            self.aggr = loaded_obj['aggr']
            self.num_epochs = loaded_obj['num_epochs']
            self.batch_percent = loaded_obj['batch_percent']
            self.learning_rate = loaded_obj['learning_rate']
            self.weight_decay = loaded_obj['weight_decay']
            self.seed = loaded_obj['seed']
            self.game_nodes = loaded_obj['game_nodes']
            self.user_nodes = loaded_obj['user_nodes']
            self.game_to_index = loaded_obj['game_to_index']
            self.user_to_index = loaded_obj['user_to_index']
            self.data = loaded_obj['data']
        # assert file_name == self.save_file_name, 'Model name in saved parameters must match the requested model.'
        self.model = HeterogeneousGraphSAGE(self.hidden_channels, self.data_metadata, self.aggr, self.num_epochs, self.batch_percent, self.learning_rate, self.weight_decay, self.seed)
        self.model.load(folder_path, f'{file_name}_{self.nn_save_name}')

    def _fine_tune(
        self,
        user_id,
        new_user_games_df,
        new_interactions_df,
        all_user_games_df,
        all_interactions_df,
        debug=False
    ):
        # print(self.data)
        if not user_id in self.user_to_index:
            self.user_to_index[user_id] = len(self.user_nodes)
            self.user_nodes.append(user_id)
            self.data['user'].x = torch.cat((self.data['user'].x, torch.ones((1, self.data['user'].x.size(1)), dtype=self.data['user'].x.dtype)), dim=0)
        if new_user_games_df.empty and new_interactions_df.empty:
            return
        # TODO Also update all the old scores (when this is actually used).
        user_indices = pd.concat([new_user_games_df['user_id'].apply(lambda id: self.user_to_index[id]), new_interactions_df['user_id'].apply(lambda id: self.user_to_index[id])])
        user_indices = torch.tensor(user_indices.values).reshape((1, -1))
        game_indices = pd.concat([new_user_games_df['game_id'].apply(lambda id: self.game_to_index[id]), new_interactions_df['game_id'].apply(lambda id: self.game_to_index[id])])
        game_indices = torch.tensor(game_indices.values).reshape((1, -1))
        scores = pd.concat([new_user_games_df['score'], new_interactions_df['score']])
        scores_tensor = torch.tensor(scores.values)
        scores_tensor = scores_tensor.type(torch.FloatTensor)
        self.data['user', 'plays', 'game'].edge_index = torch.cat((self.data['user', 'plays', 'game'].edge_index, torch.cat((user_indices, game_indices), dim=0)), dim=1)
        self.data['user', 'plays', 'game'].edge_label = torch.cat((self.data['user', 'plays', 'game'].edge_label, scores_tensor))
        self.data['game', 'rev_plays', 'user'].edge_index = torch.cat((self.data['game', 'rev_plays', 'user'].edge_index, torch.cat((game_indices, user_indices), dim=0)), dim=1)
        # print(self.data)             
