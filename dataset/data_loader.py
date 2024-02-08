from abc import ABC, abstractmethod
import copy
from sklearn.model_selection import train_test_split
from enum import Enum
import networkx as nx
import pandas as pd
import os
import numpy as np

class NodeType(Enum):
    GAME = 0
    USER = 1
    UNDEFINED_USER = 2

DATA_FILES_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data_files/')
USERS_FILENAME = 'users.csv'
GAMES_FILENAME = 'games.csv'
USERS_GAMES_FILENAME = 'users_games.csv'
FRIENDS_FILENAME = 'friends.csv'

def filter_numeric_data(data):
    def is_floatable(value):
        try:
            float(value)
            return True
        except Exception:
            return False
    return {key: float(value) for key, value in data.items() if is_floatable(value)}

def get_edges_between_types(network, node_type1, node_type2, data=False):
    nodes_type_1 = set(n for n, d in network.nodes(data=True) if d['node_type'] == node_type1)
    nodes_type_2 = set(n for n, d in network.nodes(data=True) if d['node_type'] == node_type2)
    return list(nx.edge_boundary(network, nodes_type_1, nodes_type_2, data=data))

class BaseDataLoader(ABC):
    def __init__(self):
        self.users_df = pd.read_csv(DATA_FILES_DIRECTORY + USERS_FILENAME, dtype={
            'id': 'int64',
        })
        self.games_df = pd.read_csv(DATA_FILES_DIRECTORY + GAMES_FILENAME, dtype={
            'id': 'int64',
            'name': 'string',
        })
        self.users_games_df = pd.read_csv(DATA_FILES_DIRECTORY + USERS_GAMES_FILENAME, dtype={
            'user_id': 'int64',
            'game_id': 'int64',
            # 'playtime_2weeks': 'int64',
            'playtime_forever': 'int64',
        })
        self.friends_df = pd.read_csv(DATA_FILES_DIRECTORY + FRIENDS_FILENAME, dtype={
            'user1': 'int64',
            'user2': 'int64',
        })

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
        degrees = {node: val for (node, val) in network.degree()}
        
        train_edges, test_edges = train_test_split(user_game_edges, test_size=test_percentage, train_size=train_percentage, random_state=seed, stratify=[degrees[user_node] for user_node, game_node in user_game_edges])
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
        node_strat_classes = {node: node if degree > 1 else -1 for node, degree in network.degree()}
        
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

class FriendEdgeEncoding(Enum):
    NONE = 0 # No friend edges
    BETWEEN_USERS = 1 # Will only create edges between fully defined users
    ALL_FRIENDSHIPS = 2 # Will create undefined user nodes if a node in a friendship doesn't have full data

# Default is a network with game and user nodes (hashed with ids) and edges between users and games. All options are in init.
class DataLoader(BaseDataLoader):
    def __init__(self, friendship_edge_encoding = FriendEdgeEncoding.NONE, edge_scoring_function = (lambda edge_data: 1), user_embeddings = [], game_embeddings = [], user_game_edge_embeddings = [], friend_friend_edge_embeddings = []):
        super().__init__()
        self.friendship_edge_encoding = friendship_edge_encoding
        self.user_embeddings = user_embeddings
        self.game_embeddings = game_embeddings
        self.user_game_edge_embeddings = user_game_edge_embeddings
        self.friend_friend_edge_embeddings = friend_friend_edge_embeddings
        self.edge_scoring_function = edge_scoring_function
    
    def handle_identity_embedding_command(self, network, embedding_command):
        embedding_command['add_embedding_fn'](network, embedding_command['embedding_name_base'], dict(zip(embedding_command['key'], embedding_command['args'][0])))

    def handle_sum_embedding_command(self, network, embedding_command):
        embedding_command['add_embedding_fn'](network, embedding_command['embedding_name_base'], dict(zip(embedding_command['key'], [sum(x) for x in zip(*embedding_command['args'])])))

    def dispatch_embedding_command(self, network, embedding_command):
        match embedding_command['embedding_type']:
            case EmbeddingType.IDENTITY:
                self.handle_identity_embedding_command(network, embedding_command)
            case EmbeddingType.CATEGORICAL:
                pass
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
                    command = {'embedding_type': EmbeddingType.IDENTITY, 'add_embedding_fn': add_node_embeddings, 'embedding_name_base': 'game_name', 'key': self.games_df['id'], 'args': [self.games_df['name']]}
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
            if user_game.game_id in network: # TODO remove
                network.add_edge(user_game.user_id, user_game.game_id)

        # TODO remove
        isolated_nodes = list(nx.isolates(network))
        network.remove_nodes_from(isolated_nodes)

        match self.friendship_edge_encoding:
            case FriendEdgeEncoding.NONE:
                pass
            case FriendEdgeEncoding.BETWEEN_USERS:
                for friends in self.friends_df.itertuples(index=False):
                    if friends.user1 in network.nodes() and friends.user2 in network.nodes():
                        network.add_edge(friends.user1, friends.user2)
            case FriendEdgeEncoding.ALL_FRIENDSHIPS:
                for friends in self.friends_df.itertuples(index=False):
                    if friends.user1 not in network.nodes():
                        network.add_node(friends.user1, node_type=NodeType.UNDEFINED_USER)
                    if friends.user2 not in network.nodes():
                        network.add_node(friends.user2, node_type=NodeType.UNDEFINED_USER)
                    network.add_edge(friends.user1, friends.user2)
            case _:
                raise NotImplementedError(f'Did not implement the friendship edge encoding.')
            
        self.add_embeddings(network)

        self.score_edges(network)

        return network