from abc import ABC, abstractmethod
import copy
from sklearn.model_selection import train_test_split

class BaseDataLoader(ABC):
    @abstractmethod
    def get_full_network(self):
        pass

    def uniform_train_test_network(self, train_percentage=0.9, test_percentage=0.1, seed=0):
        # TODO only remove user / game edges
        assert train_percentage + test_percentage <= 1

        network = self.get_full_network()
        edges = list(network.edges())
        
        train_edges, test_edges = train_test_split(edges, test_size=test_percentage, train_percentage=train_percentage, random_state=seed)

        train_network = copy.deepcopy(network)
        train_network.remove_edges_from([edge for edge in edges if edge not in train_edges])

        test_network = copy.deepcopy(network)
        test_network.remove_edges_from([edge for edge in edges if edge not in test_edges])

        return train_network, test_network

    def stratified_train_test_network(self, train_percentage=0.9, test_percentage=0.1, seed=0):
        # TODO only remove user / game edges
        assert train_percentage + test_percentage <= 1

        network = self.get_full_network()
        node_degrees = dict(network.degree())
        edges = list(network.edges())
        
        # TODO think about stratification method
        train_edges, test_edges = train_test_split(edges, test_size=test_percentage, train_percentage=train_percentage, random_state=seed, stratify=[min(node_degrees[edge[0]], node_degrees[edge[1]]) for edge in edges])

        train_network = copy.deepcopy(network)
        train_network.remove_edges_from([edge for edge in edges if edge not in train_edges])

        test_network = copy.deepcopy(network)
        test_network.remove_edges_from([edge for edge in edges if edge not in test_edges])

        return train_network, test_network
    
    def user_based_train_test_network(self, train_percentage=0.9, test_percentage=0.1, seed=0):
        pass