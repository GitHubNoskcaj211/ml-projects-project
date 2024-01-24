from abc import ABC, abstractmethod
import copy
from sklearn.model_selection import train_test_split
from enum import Enum
import networkx as nx
import pandas as pd
import os

class NodeType(Enum):
    GAME = 0
    USER = 1

DATA_FILES_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data_files/')
USERS_FILENAME = 'users.csv'
GAMES_FILENAME = 'games.csv'
USERS_GAMES_FILENAME = 'users_games.csv'
FRIENDS_FILENAME = 'friends.csv'

def get_edges_between_types(network, node_type1, node_type2):
    user_nodes = set(n for n, d in network.nodes(data=True) if d['bipartite'] == node_type1)
    game_nodes = set(n for n, d in network.nodes(data=True) if d['bipartite'] == node_type2)
    return list(nx.edge_boundary(network, user_nodes, game_nodes))

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

# TODO rework this to provide options for what encoding
# TODO rework this for what embedding
# Baseline - as simple as network can be. Include all user nodes and game nodes (with their id). Include user - game edges with playtime_forever attribute.
class BaselineDataLoader(BaseDataLoader):
    def __init__(self):
        super().__init__()
    
    def get_full_network(self):
        network = nx.Graph()
        # node_counter = 0
        # for index, user in self.users_df.iterrows():
        #     network.add_node(node_counter, user_id=user['id'], bipartite=NodeType.USER)
        #     node_counter += 1
        # for index, game in self.games_df.iterrows():
        #     network.add_node(node_counter, game_id=game['id'], bipartite=NodeType.GAME)
        #     node_counter += 1

        network.add_nodes_from(self.users_df.id, bipartite=NodeType.USER)
        network.add_nodes_from(self.games_df.id, bipartite=NodeType.GAME)
        # nx.set_node_attributes(network, dict(zip(self.games_df['id'], self.games_df['name'])), 'name')
        for user_game in self.users_games_df.itertuples(index=False):
            network.add_edge(user_game.user_id, user_game.game_id, playtime=user_game.playtime_forever)
            
        return network