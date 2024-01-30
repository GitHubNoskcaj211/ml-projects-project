import os
import sys
import networkx as nx
import numpy as np
import pandas as pd
from sklearn import metrics as skmetrics
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../dataset'))
from data_loader import NodeType

def absolute_error(predicted, expected):
    return abs(predicted - expected)

def squared_error(predicted, expected):
    return (predicted - expected) ** 2

# TODO relative error on embedding predictions?

def get_roc_figure(roc_curve, metric_title):
    fpr, tpr, thresholds = roc_curve
    fig, axis = plt.subplots()
    axis.plot(fpr, tpr)
    axis.scatter(fpr, tpr)
    axis.set_title(metric_title)
    axis.set_xlabel('False Positive Rate')
    axis.set_ylabel('True Positive Rate')
    return fig

class Evaluator:
    def reset(self, model):
        self.metrics = {}
        self.roc_curve = None
        self.model = model
        self.data_loader = model.data_loader
        self.test_network = self.data_loader.test_network
        self.game_nodes = set(n for n, d in self.test_network.nodes(data=True) if d['node_type'] == NodeType.GAME)
        self.all_predictions_and_scores_per_user = model.predict_for_all_users()
        edge_data = []
        for user, game_predictions in self.all_predictions_and_scores_per_user.items():
            for ii, (game, predicted_embeddings) in enumerate(game_predictions):
                expected_edge = game in self.test_network[user]
                expected_embeddings = {f'expected_{key}': value for key, value in self.test_network.get_edge_data(user, game).items()} if self.test_network.get_edge_data(user, game) is not None else {}
                predicted_embeddings = {f'predicted_{key}': value for key, value in predicted_embeddings.items()}
                edge_data.append({'user': user, 'game': game, 'user_predicted_rank': ii, 'expected_edge': expected_edge, **predicted_embeddings, **expected_embeddings})
        self.edge_results = pd.DataFrame(edge_data)

    def compute_top_N_hit_percentage(self, N):
        top_N_rows = self.edge_results[(self.edge_results['user_predicted_rank'] < N)]
        num_top_N_hits = (top_N_rows['expected_edge'] == True).sum()
        num_expected_edges = (self.edge_results['expected_edge'] == True).sum()
        self.metrics[f'top_{N}_hit_percentage'] = 1.0 if num_expected_edges == 0 else num_top_N_hits / num_expected_edges

    # Percentile is an integer 0-100.
    def compute_top_N_hit_percentage_at_user_percentile(self, N, percentile):
        filtered_rows = self.edge_results[(self.edge_results['user_predicted_rank'] < N)]
        user_top_N_hits = filtered_rows.groupby('user')['expected_edge'].sum().reset_index(name='hit_count')
        user_top_N_hits['num_expected_games'] = user_top_N_hits.apply(lambda row: len(list(nx.edge_boundary(self.test_network, [row['user']], self.game_nodes))), axis=1)
        user_top_N_hits['hit_percentage'] = user_top_N_hits['hit_count'] / user_top_N_hits['num_expected_games']
        self.metrics[f'top_{N}_hit_percentage_at_{percentile}_user_percentile'] = np.percentile(user_top_N_hits.loc[user_top_N_hits['num_expected_games'] != 0, 'hit_percentage'], percentile)
        return user_top_N_hits

    # Percentile is an integer 0-100.
    def compute_embedding_percentile_absolute_error(self, embedding, percentile):
        self.metrics[f'{embedding}_absolute_error_at_percentile_{percentile}'] = np.percentile(absolute_error(self.edge_results.loc[self.edge_results['expected_edge'] == True, f'predicted_{embedding}'], self.edge_results.loc[self.edge_results['expected_edge'] == True, f'expected_{embedding}']), percentile)
    
    # Percentile is an integer 0-100.
    def compute_score_percentile_squared_error(self, embedding, percentile):
        self.metrics[f'{embedding}_squared_error_at_percentile_{percentile}'] = np.percentile(squared_error(self.edge_results.loc[self.edge_results['expected_edge'] == True, f'predicted_{embedding}'], self.edge_results.loc[self.edge_results['expected_edge'] == True, f'expected_{embedding}']), percentile)
    
    def compute_score_mean_absolute_error(self, embedding):
        self.metrics[f'{embedding}_absolute_error_mean'] = np.mean(absolute_error(self.edge_results.loc[self.edge_results['expected_edge'] == True, f'predicted_{embedding}'], self.edge_results.loc[self.edge_results['expected_edge'] == True, f'expected_{embedding}']))
    
    def compute_score_mean_squared_error(self, embedding):
        self.metrics[f'{embedding}_squared_error_mean'] = np.mean(squared_error(self.edge_results.loc[self.edge_results['expected_edge'] == True, f'predicted_{embedding}'], self.edge_results.loc[self.edge_results['expected_edge'] == True, f'expected_{embedding}']))
    
    def plot_roc_curve(self):
        roc_curve = skmetrics.roc_curve(self.edge_results['expected_edge'].astype(int).tolist(), self.edge_results['predicted_score'])
        self.metrics['student_roc_figure'] = get_roc_figure(roc_curve, 'User-Game Predictions ROC Curve')

    def compute_auc_roc(self):
        self.metrics['auc_roc'] = skmetrics.roc_auc_score(self.edge_results['expected_edge'].astype(int).tolist(), self.edge_results['predicted_score'])