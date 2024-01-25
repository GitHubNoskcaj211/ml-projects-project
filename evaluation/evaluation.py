import os
import sys
import networkx as nx
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../dataset'))
from data_loader import NodeType

class Evaluator:
    def __init__(self):
        self.metrics = {}

    def reset_metrics(self):
        self.metrics = {}

    # TODO Handle this in base model where other models can provide specialization if it is more efficient.
    def predict_for_all_users(self, data_loader, model):
        self.all_predictions_per_user = {}
        self.test_network = data_loader.test_network
        self.game_nodes = set(n for n, d in self.test_network.nodes(data=True) if d['node_type'] == NodeType.GAME)
        for node, data in self.test_network.nodes(data=True):
            if data['node_type'] != NodeType.USER:
                continue
            self.all_predictions_per_user[node] = model.score_and_predict_n_games_for_user(node, N=None)

    # Percentile is an integer 0-100.
    def compute_top_N_hit_percentage_at_user_percentile(self, N, percentile):
        top_N_hit_percentage_per_user = {}
        for user, game_predictions in self.all_predictions_per_user.items():
            expected_games = [edge[1] for edge in nx.edge_boundary(self.test_network, [user], self.game_nodes)]
            if len(expected_games) == 0:
                continue
            top_N_hit_percentage_per_user[user] = sum((1 if game_prediction[0] in expected_games else 0 for game_prediction in game_predictions[:N])) / len(expected_games)
        self.metrics[f'top_{N}_hit_percentage_at_{percentile}_percentile'] = np.percentile(list(top_N_hit_percentage_per_user.values()), percentile)