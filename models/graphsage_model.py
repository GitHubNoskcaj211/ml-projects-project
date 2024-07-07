import sys
import os
sys.path.append(os.path.join(os.path.abspath('')))


from models.base_model import BaseGameRecommendationModel, SAVED_MODELS_PATH, SAVED_NN_PATH
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GraphConv, to_hetero
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
import random
from collections import defaultdict
from copy import deepcopy

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
        self.conv1 = GraphConv((-1, -1), hidden_channels)
        self.conv2 = GraphConv((-1, -1), hidden_channels)
        self.conv3 = GraphConv((-1, -1), out_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight=edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight=edge_weight).relu()
        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        x = self.dropout(x)

        return x


def graph_sage_neighbor_sampling(node_type_to_node_to_edge_type_to_neighbors_edge_ids_map, num_neighbors, starting_nodes_map, train_edge_percent=None):
    sampled_node_ids = defaultdict(set)
    sampled_edge_ids = defaultdict(set)
    sampled_train_edge_ids = defaultdict(set)
    max_depth = len(num_neighbors[list(num_neighbors.keys())[0]]) - 1
    queue = []
    for node_type, nodes in starting_nodes_map.items():
        queue += [(node, node_type, 0) for node in nodes]
        sampled_node_ids[node_type] |= set(nodes)
    while len(queue) > 0:
        node, node_type, depth = queue.pop(0)
        edge_type_to_neighbors_edge_ids_map = node_type_to_node_to_edge_type_to_neighbors_edge_ids_map[node_type][node]
        for edge_type, (neighbors, edge_ids) in edge_type_to_neighbors_edge_ids_map.items():
            starting_node_type, edge_name, ending_node_type = edge_type
            max_num_neighbors = num_neighbors[edge_type][depth]
            assert len(neighbors) == len(edge_ids)
            if train_edge_percent is not None and depth == 0:
                pairs = list(zip(neighbors, edge_ids))
                pairs = random.sample(pairs, int(len(neighbors) * train_edge_percent))
                if len(pairs) > 0:
                    train_neighbors, train_edge_ids = zip(*pairs)
                else:
                    train_neighbors, train_edge_ids = [], []
                neighbors = list(set(neighbors) - set(train_neighbors))
                edge_ids = list(set(edge_ids) - set(train_edge_ids))
                sampled_train_edge_ids[edge_type] |= set(train_edge_ids)
            if max_num_neighbors < len(neighbors):
                pairs = list(zip(neighbors, edge_ids))
                pairs = random.sample(pairs, max_num_neighbors)
                selected_neighbors, selected_edge_ids = zip(*pairs)
            else:
                selected_neighbors, selected_edge_ids = neighbors, edge_ids
            new_nodes = set(selected_neighbors) - sampled_node_ids[ending_node_type]
            if depth + 1 <= max_depth:
                queue += [(new_node, ending_node_type, depth + 1) for new_node in new_nodes]
            sampled_node_ids[ending_node_type] |= set(new_nodes)
            sampled_edge_ids[edge_type] |= set(selected_edge_ids)
    # print(sampled_edge_ids)
    # print(sampled_train_edge_ids)
    return sampled_node_ids, sampled_edge_ids, sampled_train_edge_ids

class CustomNeighborLoader():
    def __init__(self, data, num_neighbors, batch_size, input_nodes):
        random.seed(10)
        self.node_type_to_node_to_edge_type_to_neighbors_edge_ids_map = {}
        edge_types = list(data.edge_index_dict.keys())
        for node_type, x in data.x_dict.items():
            self.node_type_to_node_to_edge_type_to_neighbors_edge_ids_map[node_type] = {}
            matching_edge_types = [edge_type for edge_type in edge_types if edge_type[0] == node_type]
            for node in range(len(x)):
                self.node_type_to_node_to_edge_type_to_neighbors_edge_ids_map[node_type][node] = {}
                for edge_type in matching_edge_types:
                    self.node_type_to_node_to_edge_type_to_neighbors_edge_ids_map[node_type][node][edge_type] = ([], [])
        for edge_type, edge_index in data.edge_index_dict.items():
            edge_index_np = edge_index.numpy()
            num_cols = edge_index_np.shape[1]
            edge_index_np = np.vstack([edge_index_np, np.arange(num_cols).reshape(1, num_cols)])
            def add_edge(column):
                self.node_type_to_node_to_edge_type_to_neighbors_edge_ids_map[edge_type[0]][column[0]][edge_type][0].append(column[1])
                self.node_type_to_node_to_edge_type_to_neighbors_edge_ids_map[edge_type[0]][column[0]][edge_type][1].append(column[2])
            np.apply_along_axis(add_edge, 0, edge_index_np)
        self.num_neighbors = num_neighbors
        self.data = data
        self.input_nodes = input_nodes
        self.batch_size = batch_size
        
    def _get_batch(self, starting_nodes_map=None, train_edge_percent=None):
        if starting_nodes_map is None:
            starting_nodes_map = {node_type: random.sample(nodes_to_sample_from, self.batch_size) for node_type, nodes_to_sample_from in self.input_nodes.items()}
        sampled_node_ids, sampled_edge_ids, train_edge_ids = graph_sage_neighbor_sampling(self.node_type_to_node_to_edge_type_to_neighbors_edge_ids_map, self.num_neighbors, starting_nodes_map, train_edge_percent=train_edge_percent)
        data = HeteroData()
        data['user'].x = self.data['user'].x
        data['game'].x = self.data['game'].x
        data['user', 'plays', 'game'].edge_index = self.data['user', 'plays', 'game'].edge_index[:, list(sampled_edge_ids[('user', 'plays', 'game')])]
        data['game', 'rev_plays', 'user'].edge_index = self.data['game', 'rev_plays', 'user'].edge_index[:, list(sampled_edge_ids[('game', 'rev_plays', 'user')])]
        data['user', 'plays', 'game'].edge_weight = self.data['user', 'plays', 'game'].edge_weight[list(sampled_edge_ids[('user', 'plays', 'game')])]
        data['game', 'rev_plays', 'user'].edge_weight = self.data['game', 'rev_plays', 'user'].edge_weight[list(sampled_edge_ids[('game', 'rev_plays', 'user')])]
        train_edge_index = None
        train_edge_label = None
        if train_edge_percent is not None:
            train_edge_index = self.data['user', 'plays', 'game'].edge_index[:, list(train_edge_ids[('user', 'plays', 'game')])]
            train_edge_label = self.data['user', 'plays', 'game'].edge_label[list(train_edge_ids[('user', 'plays', 'game')])]
            # TODO set train_edge_label to 0 if there is no path of length 3 between the two nodes OR mask training so we dont train on those examples
            # print(train_edge_index)
            # print(train_edge_label)

        return data, starting_nodes_map, train_edge_index, train_edge_label
    
    def get_batch(self, starting_nodes_map=None):
        data, starting_nodes_map, train_edge_index, train_edge_label = self._get_batch(starting_nodes_map=starting_nodes_map)
        return data, starting_nodes_map
    
    def get_training_batch(self, train_edge_percent):
        data, starting_nodes_map, train_edge_index, train_edge_label = self._get_batch(train_edge_percent=train_edge_percent)
        return data, starting_nodes_map, train_edge_index, train_edge_label
        
# if __name__ == '__main__':
#     data = HeteroData()
#     data['user'].x = torch.full((11, 1), 1, dtype=torch.float32)
#     data['game'].x = torch.full((11, 1), 1, dtype=torch.float32)
#     data['user', 'plays', 'game'].edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10], [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 0]])
#     data['game', 'rev_plays', 'user'].edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 0], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10]])
#     data['user', 'plays', 'game'].edge_weight = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10])
#     data['game', 'rev_plays', 'user'].edge_weight = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10])
#     data['user', 'plays', 'game'].edge_label = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10])
#     loader = CustomNeighborLoader(data, {('user', 'plays', 'game'): [2, 2, 2, 1, 1], ('game', 'rev_plays', 'user'): [2, 2, 2, 1, 1]}, 3, {'user': list(range(11))})
#     batch, starting_nodes_map, train_edge_index, train_edge_label = loader.get_training_batch(0.5)
#     print(starting_nodes_map)
#     print(batch)

class MLPEdgeDecoder(torch.nn.Module):
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


class DotProductEdgeDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.sum(z_dict['user'][row] * z_dict['game'][col], axis=1)
        return z


class HeterogeneousGraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels, game_embedding_size, num_games, data_metadata, aggr, num_epochs, batch_percent, learning_rate, weight_decay, seed):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data_metadata, aggr=aggr)
        # self.decoder = MLPEdgeDecoder(hidden_channels)
        self.decoder = DotProductEdgeDecoder()
        self.embedding_game = nn.Embedding(num_games, game_embedding_size)
        self.num_epochs = num_epochs
        self.batch_percent = batch_percent
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loader = None

        self.new_seed(seed)

        self.loss_fn = nn.L1Loss()

    def new_seed(self, seed=None):
        if seed is None:
            torch.seed()
        else:
            torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed

    def forward(self, x_dict, edge_index_dict, edge_weight_dict, edge_score_index):
        # TODO Does this concatenate multiple times on successive calls?
        x_dict['game'] = torch.cat((x_dict['game'], self.embedding_game.weight), axis=1)
        z_dict = self.encoder(x_dict, edge_index_dict, edge_weight_dict)
        return self.decoder(z_dict, edge_score_index)

    def get_batch_for_users(self, train_data, user_indices=None):
        if user_indices is None:
            user_indices = list(range(len(train_data.x_dict['user'])))
        batch, starting_nodes_map = self.loader.get_batch(starting_nodes_map={'user': user_indices})
        return batch

    def train(self, train_data, test_edge_index, test_labels, optimal_model_save_name, last_model_save_name, debug=False, writer=None):
        super().train(True)
        # Initializes the network.
        self.forward(train_data.x_dict, {key: value[:, 0:1] for key, value in train_data.edge_index_dict.items()}, {key: value[0:1] for key, value in train_data.edge_weight_dict.items()},
                    train_data['user', 'game'].edge_index[:, 0:1])

        for p in self.parameters():
            p.requires_grad_(True)
        
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)
        print('Total Learnable Parameters:', sum(p.numel() for p in self.parameters() if p.requires_grad))
        batch_size = int(len(train_data['user']['x']) * self.batch_percent) + 1
        if self.loader is None:
            self.loader = CustomNeighborLoader(train_data, {key: [50, 25, 10] for key in train_data.edge_types}, batch_size, {'user': list(range(len(train_data.x_dict['user'])))})

        train_loss = []
        lowest_test_loss = None
        for epoch_count in tqdm(range(self.num_epochs), desc='Training'):
            epoch_loss = []
            percent_data_batched = 0
            while percent_data_batched < 1:
                batch, starting_nodes_map, train_edge_index, train_edge_label = self.loader.get_training_batch(0.2)
                pred = self.forward(batch.x_dict, batch.edge_index_dict, batch.edge_weight_dict, train_edge_index)
                # TODO We perform a new round of negative sampling for every training epoch?
                loss = self.loss_fn(pred, train_edge_label)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
                percent_data_batched += self.batch_percent
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
        if self.loader is None:
            self.loader = CustomNeighborLoader(train_data, {key: [50, 25, 10] for key in train_data.edge_types}, None, {'user': list(range(len(train_data.x_dict['user'])))})
        batch = self.get_batch_for_users(train_data, user_indices=None)
        pred = self.forward(batch.x_dict, batch.edge_index_dict, batch.edge_weight_dict, test_edge_index)
        loss = self.loss_fn(pred, test_labels)
        super().train(True)
        return loss.item()

    @torch.no_grad()
    def predict(self, train_data, user_indices, edge_score_index):
        super().train(False)
        # TODO How does predict work if the game node is not connected anywhere in the subgraph... should just be 0?
        if self.loader is None:
            self.loader = CustomNeighborLoader(train_data, {key: [50, 25, 10] for key in train_data.edge_types}, None, {'user': list(range(len(train_data.x_dict['user'])))})
        batch = self.get_batch_for_users(train_data, user_indices)
        output = self.forward(batch.x_dict, batch.edge_index_dict, batch.edge_weight_dict, edge_score_index)
        return output.detach().flatten().tolist()

    def _save(self, file_name, overwrite=False):
        assert not os.path.isfile(SAVED_MODELS_PATH + SAVED_NN_PATH + file_name + '.pth') or overwrite, f'Tried to save to a file that already exists {file_name} without allowing for overwrite.'
        torch.save(self.state_dict(), os.path.join(SAVED_MODELS_PATH, SAVED_NN_PATH, file_name + '.pth'))

    def load(self, folder_path, file_name):
        self.load_state_dict(torch.load(os.path.join(folder_path, SAVED_NN_PATH, file_name + '.pth')))


class GraphSAGE(BaseGameRecommendationModel):
    def __init__(self, hidden_channels=50, game_embedding_size=50, aggr='mean', save_file_name=None, nn_save_name=None, num_epochs=50, batch_percent=0.1, learning_rate=5e-3, weight_decay=1e-5, seed=None):
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
        self.game_embedding_size = game_embedding_size

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
        game_tensor = known_game_embeddings_tensor
        self.data['game'].x = game_tensor

        user_indices = torch.tensor(train_users_games_df['user_id'].apply(lambda id: self.user_to_index[id]).values).reshape((1, -1))
        game_indices = torch.tensor(train_users_games_df['game_id'].apply(lambda id: self.game_to_index[id]).values).reshape((1, -1))
        self.data['user', 'plays', 'game'].edge_index = torch.cat((user_indices, game_indices), dim=0)
        user_game_scores_tensor = torch.tensor(train_users_games_df['score'].values)
        user_game_scores_tensor = user_game_scores_tensor.type(torch.FloatTensor)
        self.data['user', 'plays', 'game'].edge_weight = user_game_scores_tensor
        self.data['user', 'plays', 'game'].edge_label = user_game_scores_tensor

        self.data = T.ToUndirected()(self.data)
        del self.data['game', 'rev_plays', 'user'].edge_label

        self.data_metadata = self.data.metadata()
        self.model = HeterogeneousGraphSAGE(self.hidden_channels, self.game_embedding_size, len(self.game_nodes), self.data_metadata, self.aggr, self.num_epochs, self.batch_percent, self.learning_rate, self.weight_decay, self.seed)
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
                'game_embedding_size': self.game_embedding_size,
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
            self.game_embedding_size = loaded_obj['game_embedding_size']
            self.game_nodes = loaded_obj['game_nodes']
            self.user_nodes = loaded_obj['user_nodes']
            self.game_to_index = loaded_obj['game_to_index']
            self.user_to_index = loaded_obj['user_to_index']
            self.data = loaded_obj['data']
        # assert file_name == self.save_file_name, 'Model name in saved parameters must match the requested model.'
        self.model = HeterogeneousGraphSAGE(self.hidden_channels, self.game_embedding_size, len(self.game_nodes), self.data_metadata, self.aggr, self.num_epochs, self.batch_percent, self.learning_rate, self.weight_decay, self.seed)
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
        self.data['user', 'plays', 'game'].edge_weight = torch.cat((self.data['user', 'plays', 'game'].edge_weight, scores_tensor))
        self.data['game', 'rev_plays', 'user'].edge_weight = torch.cat((self.data['game', 'rev_plays', 'user'].edge_weight, scores_tensor))
        self.data['game', 'rev_plays', 'user'].edge_index = torch.cat((self.data['game', 'rev_plays', 'user'].edge_index, torch.cat((game_indices, user_indices), dim=0)), dim=1)
        # print(self.data)             
