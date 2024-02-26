from abc import ABC, abstractmethod
import copy
from sklearn.model_selection import train_test_split
from enum import Enum
import networkx as nx
import pandas as pd
import os
import numpy as np
from pprint import pprint
import dill
import pickle
from ast import literal_eval

import sys
sys.path.append("../utils/")
from utils import linear_transformation, gaussian_transformation

class LogType(Enum):
    ADD_QUEUE = 1
    VISITED_VALID = 2
    VISITED_INVALID = 3

SAVED_DATA_LOADER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_data_loader_parameters/')

class NodeType(Enum):
    GAME = 0
    USER = 1
    UNDEFINED_USER = 2

class FriendEdgeEncoding(Enum):
    NONE = 0 # No friend edges
    BETWEEN_USERS = 1 # Will only create edges between fully defined users
    ALL_FRIENDSHIPS = 2 # Will create undefined user nodes if a node in a friendship doesn't have full data

DATA_FILES_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data_files/')
USERS_FILENAME = 'users.csv'
GAMES_FILENAME = 'games.csv'
USERS_GAMES_FILENAME = 'users_games.csv'
FRIENDS_FILENAME = 'friends.csv'

def print_game_edges_for_user(network, user):
    print(f'Edges for user {user}:')
    game_nodes = [node for node, data in network.nodes(data=True) if data['node_type'] == NodeType.GAME]
    edges = nx.edge_boundary(network, [user], game_nodes, data=True)
    pprint(list(edges))

def get_edges_between_types(network, node_type1, node_type2, data=False):
    nodes_type_1 = set(n for n, d in network.nodes(data=True) if d['node_type'] == node_type1)
    nodes_type_2 = set(n for n, d in network.nodes(data=True) if d['node_type'] == node_type2)
    return list(nx.edge_boundary(network, nodes_type_1, nodes_type_2, data=data))

def read_log_file(file_path):
    result = []
    with open(file_path, 'r') as file:
        for line in file:
            int_list = list(map(int, line.split()))
            result.append(int_list)
    return result

class BaseDataLoader(ABC):
    @abstractmethod
    def get_full_network(self):
        pass

    def load_random_train_test_network(self, network=None, train_percentage=0.9, test_percentage=0.1, seed=0):
        assert train_percentage + test_percentage <= 1
        
        if network is None:
            network = self.get_full_network()
        user_game_edges = get_edges_between_types(network, NodeType.USER, NodeType.GAME)

        train_edges, test_edges = train_test_split(user_game_edges, test_size=test_percentage, train_size=train_percentage, random_state=seed)
        user_game_edges_set = set(user_game_edges)
        self.train_network = copy.deepcopy(network)
        self.train_network.remove_edges_from(user_game_edges_set - set(train_edges))
        self.test_network = copy.deepcopy(network)
        self.test_network.remove_edges_from(user_game_edges_set - set(test_edges))

    def load_stratified_user_degree_train_test_network(self, network=None, train_percentage=0.9, test_percentage=0.1, seed=0):
        assert train_percentage + test_percentage <= 1

        if network is None:
            network = self.get_full_network()
        user_game_edges = get_edges_between_types(network, NodeType.USER, NodeType.GAME)
        game_nodes = [node for node, data in network.nodes(data=True) if data['node_type'] == NodeType.GAME]
        game_degrees = {node: len(list(nx.edge_boundary(network, [node], game_nodes, data=True))) for node, data in network.nodes(data=True) if data['node_type'] == NodeType.USER}
        
        train_edges, test_edges = train_test_split(user_game_edges, test_size=test_percentage, train_size=train_percentage, random_state=seed, stratify=[game_degrees[user_node] for user_node, game_node in user_game_edges])
        user_game_edges_set = set(user_game_edges)
        self.train_network = copy.deepcopy(network)
        self.train_network.remove_edges_from(user_game_edges_set - set(train_edges))
        self.test_network = copy.deepcopy(network)
        self.test_network.remove_edges_from(user_game_edges_set - set(test_edges))
    
    def load_stratified_user_train_test_network(self, network=None, train_percentage=0.9, test_percentage=0.1, seed=0):
        assert train_percentage + test_percentage <= 1

        if network is None:
            network = self.get_full_network()
        user_game_edges = get_edges_between_types(network, NodeType.USER, NodeType.GAME)
        # If any nodes have 1 or 0 edges - remap them to an "other" stratification grouping.
        game_nodes = [node for node, data in network.nodes(data=True) if data['node_type'] == NodeType.GAME]
        node_strat_classes = {node: node if len(list(nx.edge_boundary(network, [node], game_nodes, data=True))) > 1 else -1 for node, data in network.nodes(data=True) if data['node_type'] == NodeType.USER}

        train_edges, test_edges = train_test_split(user_game_edges, test_size=test_percentage, train_size=train_percentage, random_state=seed, stratify=[node_strat_classes[user_node] for user_node, game_node in user_game_edges])
        user_game_edges_set = set(user_game_edges)
        self.train_network = copy.deepcopy(network)
        self.train_network.remove_edges_from(user_game_edges_set - set(train_edges))
        self.test_network = copy.deepcopy(network)
        self.test_network.remove_edges_from(user_game_edges_set - set(test_edges))

class EmbeddingType(Enum):
    IDENTITY = 0
    CATEGORICAL = 1
    ONE_HOT = 2
    SUM = 3

def add_node_embeddings(network, embedding_name, node_to_value_dict):
    nx.set_node_attributes(network, node_to_value_dict, embedding_name)

def add_edge_embeddings(network, embedding_name, edge_to_value_dict):
    nx.set_edge_attributes(network, edge_to_value_dict, embedding_name)

class Normalizer(ABC):
    @abstractmethod
    def normalize(self, network):
        pass

class LinearNormalizer(Normalizer):
    def __init__(self, min_score, max_score):
        self.min_score = min_score
        self.max_score = max_score
    
    def normalize(self, network):
        min_max_score_per_user = {}
        game_nodes = [node for node, data in network.nodes(data=True) if data['node_type'] == NodeType.GAME]
        for node, data in network.nodes(data=True):
            if data['node_type'] != NodeType.USER:
                continue
            edges = nx.edge_boundary(network, [node], game_nodes, data=True)
            scores = [data['score'] for user, game, data in edges]
            min_max_score_per_user[node] = (min(scores), max(scores))

        for edge in get_edges_between_types(network, NodeType.USER, NodeType.GAME):
            user = edge[0]
            network.edges[edge]['score'] = linear_transformation(network.edges[edge]['score'], *min_max_score_per_user[user], self.min_score, self.max_score)

class GaussianNormalizer(Normalizer):
    def __init__(self, new_mean, new_std):
        self.new_mean = new_mean
        self.new_std = new_std
    
    def normalize(self, network):
        mean_stddev_score_per_user = {}
        game_nodes = [node for node, data in network.nodes(data=True) if data['node_type'] == NodeType.GAME]
        for node, data in network.nodes(data=True):
            if data['node_type'] != NodeType.USER:
                continue
            edges = nx.edge_boundary(network, [node], game_nodes, data=True)
            scores = [data['score'] for user, game, data in edges]
            if len(scores) > 0:
                mean_stddev_score_per_user[node] = (np.mean(scores), np.std(scores))

        for edge in get_edges_between_types(network, NodeType.USER, NodeType.GAME):
            user = edge[0]
            network.edges[edge]['score'] = gaussian_transformation(network.edges[edge]['score'], *mean_stddev_score_per_user[user], self.new_mean, self.new_std)

class PercentileNormalizer(Normalizer):
    def normalize(self, network):
        game_nodes = [node for node, data in network.nodes(data=True) if data['node_type'] == NodeType.GAME]
        for node, data in network.nodes(data=True):
            if data['node_type'] != NodeType.USER:
                continue
            edges = list(nx.edge_boundary(network, [node], game_nodes, data=True))
            scores = np.array([data['score'] for user, game, data in edges])
            for user, game, data in edges:
                network.edges[(user, game)]['score'] = (scores <= data['score']).mean()

def constant_edge_scoring_function(edge_data):
    return 1

def playtime_forever_edge_scoring_function(edge_data):
    return edge_data['playtime_forever']

def never_remove_edge(edge_data):
    return False

def remove_zero_playtime_edge(edge_data):
    return edge_data['playtime_forever'] == 0

# Default is a network with game and user nodes (hashed with ids) and edges between users and games. All options are in init.
class DataLoader(BaseDataLoader):
    def __init__(self, friendship_edge_encoding = FriendEdgeEncoding.NONE, edge_scoring_function = constant_edge_scoring_function, score_normalizers = [], user_embeddings = [], game_embeddings = [], user_game_edge_embeddings = [], friend_friend_edge_embeddings = [], snowballs_ids = [], num_users_to_load_per_snowball = None, remove_edge_function = never_remove_edge):
        super().__init__()
        self.friendship_edge_encoding = friendship_edge_encoding
        self.edge_scoring_function = edge_scoring_function
        self.score_normalizers = score_normalizers
        self.user_embeddings = user_embeddings
        self.game_embeddings = game_embeddings
        self.user_game_edge_embeddings = user_game_edge_embeddings
        self.friend_friend_edge_embeddings = friend_friend_edge_embeddings
        self.snowball_ids = snowballs_ids
        self.num_users_to_load_per_snowball = num_users_to_load_per_snowball
        if len(self.snowball_ids) == 0 and self.num_users_to_load_per_snowball is not None:
                self.snowball_ids = [subfolder for subfolder in os.listdir(DATA_FILES_DIRECTORY) if os.path.isdir(os.path.join(DATA_FILES_DIRECTORY, subfolder))]
        self.remove_edge_function = remove_edge_function
        self.load_data_files()

    def load_data_files(self):
        global_users_df = pd.read_csv(DATA_FILES_DIRECTORY + USERS_FILENAME, dtype={
            'id': 'int64',
        })
        global_games_df = pd.read_csv(DATA_FILES_DIRECTORY + GAMES_FILENAME, dtype={
            'id': 'int64',
            'name': 'string',
            'numReviews': 'int64',
            'avgReviewScore': 'int64',
            'price': 'float64',
            'numFollowers': 'int64',
        }, converters={'genres': literal_eval, 'tags': literal_eval})
        global_users_games_df = pd.read_csv(DATA_FILES_DIRECTORY + USERS_GAMES_FILENAME, dtype={
            'user_id': 'int64',
            'game_id': 'int64',
            # 'playtime_2weeks': 'int64',
            'playtime_forever': 'int64',
        })
        global_friends_df = pd.read_csv(DATA_FILES_DIRECTORY + FRIENDS_FILENAME, dtype={
            'user1': 'int64',
            'user2': 'int64',
        })

        if len(self.snowball_ids) == 0 and self.num_users_to_load_per_snowball is None:
            self.users_df = global_users_df
            self.games_df = global_games_df
            self.users_games_df = global_users_games_df
            self.friends_df = global_friends_df
        else:       
            users_dfs = []
            games_dfs = []
            for snowball_id in self.snowball_ids:
                users_snowball_df = pd.read_csv(DATA_FILES_DIRECTORY + snowball_id + '/' + USERS_FILENAME, usecols=['id'], dtype={
                    'id': 'int64',
                })
                if self.num_users_to_load_per_snowball is not None:
                    users_snowball_df = users_snowball_df.iloc[:self.num_users_to_load_per_snowball]
                games_snowball_df = pd.read_csv(DATA_FILES_DIRECTORY + snowball_id + '/' + GAMES_FILENAME, usecols=['id'], dtype={
                    'id': 'int64'
                })
                users_dfs.append(users_snowball_df)
                games_dfs.append(games_snowball_df)

            users_df = pd.concat(users_dfs, ignore_index=True)
            games_df = pd.concat(games_dfs, ignore_index=True)

            self.users_df = global_users_df[global_users_df['id'].isin(users_df['id'])]
            self.games_df = global_games_df[global_games_df['id'].isin(games_df['id'])]
            self.users_games_df = global_users_games_df[
                (global_users_games_df['user_id'].isin(users_df['id'])) &
                (global_users_games_df['game_id'].isin(games_df['id']))
            ]
            self.friends_df = global_friends_df[
                (global_friends_df['user1'].isin(users_df['id'])) |
                (global_friends_df['user2'].isin(users_df['id']))
            ]
            

    @classmethod
    def load_from_file(cls, file_name):
        with open(SAVED_DATA_LOADER_PATH + file_name + '.pkl', 'rb') as file:
            parameter_dictionary = pickle.load(file)
            return cls(parameter_dictionary['friendship_edge_encoding'],
                parameter_dictionary['edge_scoring_function'],
                parameter_dictionary['score_normalizers'],
                parameter_dictionary['user_embeddings'],
                parameter_dictionary['game_embeddings'],
                parameter_dictionary['user_game_edge_embeddings'],
                parameter_dictionary['friend_friend_edge_embeddings'],
                parameter_dictionary['snowballs_ids'],
                parameter_dictionary['num_users_to_load_per_snowball'],
                parameter_dictionary['remove_edge_function'])
    
    def get_data_loader_parameters(self):
        return {
            'friendship_edge_encoding': self.friendship_edge_encoding,
            'edge_scoring_function': self.edge_scoring_function,
            'score_normalizers': self.score_normalizers,
            'user_embeddings': self.user_embeddings,
            'game_embeddings': self.game_embeddings,
            'user_game_edge_embeddings': self.user_game_edge_embeddings,
            'friend_friend_edge_embeddings': self.friend_friend_edge_embeddings,
            'snowballs_ids': self.snowballs_ids,
            'num_users_to_load_per_snowball': self.num_users_to_load_per_snowball,
            'remove_edge_function': self.remove_edge_function,
        }
    
    def save_data_loader_parameters(self, file_name, overwrite=False):
        assert not os.path.isfile(SAVED_DATA_LOADER_PATH + file_name + '.pkl') or overwrite, f'Tried to save to a file that already exists {file_name} without allowing for overwrite.'
        with open(SAVED_DATA_LOADER_PATH + file_name + '.pkl', 'wb') as file:
            pickle.dump(self.get_data_loader_parameters(), file)
    
    def handle_identity_embedding_command(self, network, embedding_command):
        embedding_command['add_embedding_fn'](network, embedding_command['embedding_name_base'], dict(zip(embedding_command['key'], embedding_command['args'][0])))

    def handle_sum_embedding_command(self, network, embedding_command):
        embedding_command['add_embedding_fn'](network, embedding_command['embedding_name_base'], dict(zip(embedding_command['key'], [sum(x) for x in zip(*embedding_command['args'])])))

    def handle_categorical_embedding_command(self, network, embedding_command):
        categories = set([value for lst in embedding_command['args'][0] for value in lst])
        for category in categories:
            embedding_command['add_embedding_fn'](network, embedding_command['embedding_name_base'] + ': ' + category, dict(zip(embedding_command['key'], embedding_command['args'][0].apply(lambda lst: 1.0 if category in lst else 0.0))))

    def dispatch_embedding_command(self, network, embedding_command):
        match embedding_command['embedding_type']:
            case EmbeddingType.IDENTITY:
                self.handle_identity_embedding_command(network, embedding_command)
            case EmbeddingType.CATEGORICAL:
                self.handle_categorical_embedding_command(network, embedding_command)
            case EmbeddingType.ONE_HOT:
                pass
            case EmbeddingType.SUM:
                self.handle_sum_embedding_command(network, embedding_command)
            case _:
                raise NotImplementedError(f'Have embedding type that cannot be handled: {embedding_command["embedding_type"]}')

    def run_user_embedding_commands(self, network, user_embeddings):
        for user_embedding in user_embeddings:
            match user_embedding:
                case _:
                    raise NotImplementedError(f'Cannot recognize embedding: {user_embedding}')
            self.dispatch_embedding_command(network, command)
    
    def run_game_embedding_commands(self, network, game_embeddings):
        for game_embedding in game_embeddings:
            match game_embedding:
                case 'name':
                    command = {'embedding_type': EmbeddingType.IDENTITY, 'add_embedding_fn': add_node_embeddings, 'embedding_name_base': 'name', 'key': self.games_df['id'], 'args': [self.games_df['name']]}
                case 'numReviews':
                    command = {'embedding_type': EmbeddingType.IDENTITY, 'add_embedding_fn': add_node_embeddings, 'embedding_name_base': 'num_reviews', 'key': self.games_df['id'], 'args': [self.games_df['numReviews']]}
                case 'avgReviewScore':
                    command = {'embedding_type': EmbeddingType.IDENTITY, 'add_embedding_fn': add_node_embeddings, 'embedding_name_base': 'avg_review_score', 'key': self.games_df['id'], 'args': [self.games_df['avgReviewScore']]}
                case 'price':
                    command = {'embedding_type': EmbeddingType.IDENTITY, 'add_embedding_fn': add_node_embeddings, 'embedding_name_base': 'price', 'key': self.games_df['id'], 'args': [self.games_df['price']]}
                case 'genres':
                    command = {'embedding_type': EmbeddingType.CATEGORICAL, 'add_embedding_fn': add_node_embeddings, 'embedding_name_base': 'Genre', 'key': self.games_df['id'], 'args': [self.games_df['genres']]}
                case 'tags':
                    command = {'embedding_type': EmbeddingType.CATEGORICAL, 'add_embedding_fn': add_node_embeddings, 'embedding_name_base': 'Tag', 'key': self.games_df['id'], 'args': [self.games_df['tags']]}
                case 'numFollowers':
                    command = {'embedding_type': EmbeddingType.IDENTITY, 'add_embedding_fn': add_node_embeddings, 'embedding_name_base': 'num_followers', 'key': self.games_df['id'], 'args': [self.games_df['numFollowers']]}
                case _:
                    raise NotImplementedError(f'Cannot recognize embedding: {game_embedding}')
            self.dispatch_embedding_command(network, command)

    def run_user_game_edge_embedding_commands(self, network, user_game_edge_embeddings):
        for user_game_edge_embedding in user_game_edge_embeddings:
            match user_game_edge_embedding:
                case 'example_sum_user_id_game_id_playtime_forever':
                    command = {'embedding_type': EmbeddingType.SUM, 'add_embedding_fn': add_edge_embeddings, 'embedding_name_base': 'example_sum_user_id_game_id_playtime_forever', 'key': ((u, g) for u, g in zip(self.users_games_df['user_id'], self.users_games_df['game_id'])), 'args': [self.users_games_df['user_id'], self.users_games_df['game_id'], self.users_games_df['playtime_forever']]}
                case 'playtime_forever':
                    command = {'embedding_type': EmbeddingType.IDENTITY, 'add_embedding_fn': add_edge_embeddings, 'embedding_name_base': 'playtime_forever', 'key': ((u, g) for u, g in zip(self.users_games_df['user_id'], self.users_games_df['game_id'])), 'args': [self.users_games_df['playtime_forever']]}
                case _:
                    raise NotImplementedError(f'Cannot recognize embedding: {user_game_edge_embedding}')
            self.dispatch_embedding_command(network, command)

    def run_friend_friend_edge_embedding_commands(self, network, friend_friend_edge_embeddings):
        for friend_friend_edge_embedding in friend_friend_edge_embeddings:
            match friend_friend_edge_embedding:
                case _:
                    raise NotImplementedError(f'Cannot recognize embedding: {friend_friend_edge_embedding}')
            self.dispatch_embedding_command(network, command)

    def remove_edges(self, network):
        edges_to_remove = [edge for edge in get_edges_between_types(network, NodeType.USER, NodeType.GAME) if self.remove_edge_function(network.edges[edge])]
        network.remove_edges_from(edges_to_remove)

    def score_edges(self, network):
        for edge in get_edges_between_types(network, NodeType.USER, NodeType.GAME):
            edge_data = network.edges[edge]
            score = self.edge_scoring_function(edge_data)
            network.edges[edge]['score'] = score

    def add_embeddings(self, network):
        self.run_user_embedding_commands(network, self.user_embeddings)
        self.run_game_embedding_commands(network, self.game_embeddings)
        self.run_user_game_edge_embedding_commands(network, self.user_game_edge_embeddings)
        self.run_friend_friend_edge_embedding_commands(network, self.friend_friend_edge_embeddings)

    def get_full_network(self):
        network = nx.Graph()
        network.add_nodes_from(self.users_df.id, node_type=NodeType.USER)                
        network.add_nodes_from(self.games_df.id, node_type=NodeType.GAME)

        for user_game in self.users_games_df.itertuples(index=False):
            if user_game.game_id in network and user_game.user_id in network:
                network.add_edge(user_game.user_id, user_game.game_id)
            else:
                continue
                print(f'Something is broken. {user_game.game_id} {user_game.user_id}') # TODO Add this back in after Akash fixes gamalytic data missing.

        isolated_nodes = list(nx.isolates(network))
        network.remove_nodes_from(isolated_nodes)

        if self.friendship_edge_encoding == FriendEdgeEncoding.NONE:
            pass
        elif self.friendship_edge_encoding == FriendEdgeEncoding.BETWEEN_USERS:
            for friends in self.friends_df.itertuples(index=False):
                if friends.user1 in network.nodes() and friends.user2 in network.nodes():
                    network.add_edge(friends.user1, friends.user2)
        elif self.friendship_edge_encoding == FriendEdgeEncoding.ALL_FRIENDSHIPS:
            for friends in self.friends_df.itertuples(index=False):
                if friends.user1 not in network.nodes():
                    network.add_node(friends.user1, node_type=NodeType.UNDEFINED_USER)
                if friends.user2 not in network.nodes():
                    network.add_node(friends.user2, node_type=NodeType.UNDEFINED_USER)
                network.add_edge(friends.user1, friends.user2)
            
        self.add_embeddings(network)

        self.remove_edges(network)

        self.score_edges(network)

        for normalizer in self.score_normalizers:
            normalizer.normalize(network)

        return network