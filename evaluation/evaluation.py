import os
import sys
import networkx as nx
import numpy as np
import pandas as pd
from sklearn import metrics as skmetrics
from matplotlib import pyplot as plt
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../dataset'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utils'))
from data_loader import NodeType, get_edges_between_types
from utils import linear_transformation

SAVED_EVALUATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_evaluation/')

def absolute_error(predicted, expected):
    return abs(predicted - expected)

def squared_error(predicted, expected):
    return (predicted - expected) ** 2

def get_roc_figure(roc_curve, metric_title):
    fpr, tpr, thresholds = roc_curve
    fig, axis = plt.subplots()
    axis.plot(fpr, tpr)
    axis.scatter(fpr, tpr)
    axis.set_title(metric_title)
    axis.set_xlabel('False Positive Rate')
    axis.set_ylabel('True Positive Rate')
    return fig

def get_percentile_figure(values, metric_title):
    sorted_values = sorted(values)
    fig, axis = plt.subplots()
    percentiles = [i / (len(sorted_values) - 1) for i in range(len(sorted_values))]
    axis.plot(percentiles, sorted_values)
    axis.set_title(metric_title)
    axis.set_xlabel('User Percentiles')
    axis.set_ylabel('Metric Value')
    return fig

class Evaluator:
    def __init__(self, top_N_games_to_eval):
        self.top_N_games_to_eval = top_N_games_to_eval

    def reset(self, model, debug=False):
        if debug:
            print(model.name())
        self.metrics = {}
        self.roc_curve = None
        self.model = model
        self.data_loader = model.data_loader
        self.test_network = self.data_loader.test_network
        self.game_nodes = set(n for n, d in self.test_network.nodes(data=True) if d['node_type'] == NodeType.GAME)
        self.all_predictions_and_scores_per_user = model.predict_for_all_users(N = self.top_N_games_to_eval)
        if debug:
            print('Done getting predictions.')

        # TODO Don't have to have scores sorted for this - might make eval faster.
        edge_data = []
        for user, game_predictions in self.all_predictions_and_scores_per_user.items():
            for game, predicted_score in game_predictions:
                expected_edge = game in self.test_network[user]
                expected_score = self.test_network.get_edge_data(user, game)['score'] if expected_edge else None
                edge_data.append({'user': user, 'game': game, 'expected_edge': expected_edge, 'predicted_score': predicted_score, 'expected_score': expected_score})
        self.top_N_edge_results = pd.DataFrame(edge_data)
        
        expected_missed_edge_data = []
        for user, game, data in get_edges_between_types(self.test_network, NodeType.USER, NodeType.GAME, data=True):
            if not self.top_N_edge_results[(self.top_N_edge_results['user'] == user) & (self.top_N_edge_results['game'] == game)].empty:
                continue
            expected_edge = game in self.test_network[user]
            expected_missed_edge_data.append({'user': user, 'game': game, 'expected_edge': expected_edge, 'predicted_score': model.get_score_between_user_and_game(user, game), 'expected_score': data['score']})
        
        self.top_N_and_all_expected_edge_results = pd.concat([self.top_N_edge_results, pd.DataFrame(expected_missed_edge_data)], ignore_index = True) 
        self.top_N_edge_results['user_predicted_rank'] = self.top_N_edge_results.groupby('user')['predicted_score'].rank(ascending=False) - 1
        self.top_N_and_all_expected_edge_results['user_predicted_rank'] = self.top_N_and_all_expected_edge_results.groupby('user')['predicted_score'].rank(ascending=False) - 1

        if debug:
            print('Done getting edge results.')
        self.positional_error_scores = None

    def compute_top_N_hit_percentage(self, N):
        assert N < self.top_N_games_to_eval, 'Cannot get top N hit percentage when we have less top N games to eval since the dataframe is malformed.'
        top_N_rows = self.top_N_and_all_expected_edge_results[(self.top_N_and_all_expected_edge_results['user_predicted_rank'] < N)]
        num_top_N_hits = (top_N_rows['expected_edge'] == True).sum()
        num_expected_edges = (self.top_N_and_all_expected_edge_results['expected_edge'] == True).sum()
        self.metrics[f'top_{N}_hit_percentage'] = 1.0 if num_expected_edges == 0 else num_top_N_hits / num_expected_edges

    def get_top_N_hit_percentage_per_user(self, N):
        assert N < self.top_N_games_to_eval, 'Cannot get top N hit percentage when we have less top N games to eval since the dataframe is malformed.'
        filtered_rows = self.top_N_and_all_expected_edge_results[(self.top_N_and_all_expected_edge_results['user_predicted_rank'] < N)]
        user_top_N_hits = filtered_rows.groupby('user')['expected_edge'].sum().reset_index(name='hit_count')
        user_top_N_hits['num_expected_games'] = user_top_N_hits.apply(lambda row: len(list(nx.edge_boundary(self.test_network, [row['user']], self.game_nodes))), axis=1)
        user_top_N_hits['hit_percentage'] = user_top_N_hits['hit_count'] / user_top_N_hits['num_expected_games']
        return user_top_N_hits

    def plot_top_N_hit_percentage_percentiles(self, N):
        user_top_N_hits = self.get_top_N_hit_percentage_per_user(N)
        self.metrics[f'top_{N}_hit_percentage_user_percentiles_figure'] = get_percentile_figure(user_top_N_hits.loc[user_top_N_hits['num_expected_games'] != 0, 'hit_percentage'].values, f'User Top {N} Hit Percentage Percentiles')
    
    # Percentile is an integer 0-100.
    def compute_top_N_hit_percentage_at_user_percentile(self, N, percentile):
        user_top_N_hits = self.get_top_N_hit_percentage_per_user(N)
        self.metrics[f'top_{N}_hit_percentage_at_{percentile}_user_percentile'] = np.percentile(user_top_N_hits.loc[user_top_N_hits['num_expected_games'] != 0, 'hit_percentage'], percentile)

    # Percentile is an integer 0-100.
    def compute_embedding_percentile_absolute_error(self, embedding, percentile):
        self.metrics[f'{embedding}_absolute_error_at_percentile_{percentile}'] = np.percentile(absolute_error(self.top_N_and_all_expected_edge_results.loc[self.top_N_and_all_expected_edge_results['expected_edge'] == True, f'predicted_{embedding}'], self.top_N_and_all_expected_edge_results.loc[self.top_N_and_all_expected_edge_results['expected_edge'] == True, f'expected_{embedding}']), percentile)
    
    # Percentile is an integer 0-100.
    def compute_embedding_percentile_squared_error(self, embedding, percentile):
        self.metrics[f'{embedding}_squared_error_at_percentile_{percentile}'] = np.percentile(squared_error(self.top_N_and_all_expected_edge_results.loc[self.top_N_and_all_expected_edge_results['expected_edge'] == True, f'predicted_{embedding}'], self.top_N_and_all_expected_edge_results.loc[self.top_N_and_all_expected_edge_results['expected_edge'] == True, f'expected_{embedding}']), percentile)
    
    def plot_embedding_percentile_absolute_error(self, embedding):
        self.metrics[f'{embedding}_absolute_error_percentiles_figure'] = get_percentile_figure(absolute_error(self.top_N_and_all_expected_edge_results.loc[self.top_N_and_all_expected_edge_results['expected_edge'] == True, f'predicted_{embedding}'], self.top_N_and_all_expected_edge_results.loc[self.top_N_and_all_expected_edge_results['expected_edge'] == True, f'expected_{embedding}']).values, f'Embedding {embedding} Absolute Error at Percentiles')
    
    def plot_embedding_percentile_squared_error(self, embedding):
        self.metrics[f'{embedding}_squared_error_percentiles_figure'] = get_percentile_figure(squared_error(self.top_N_and_all_expected_edge_results.loc[self.top_N_and_all_expected_edge_results['expected_edge'] == True, f'predicted_{embedding}'], self.top_N_and_all_expected_edge_results.loc[self.top_N_and_all_expected_edge_results['expected_edge'] == True, f'expected_{embedding}']).values, f'Embedding {embedding} Squared Error at Percentiles')

    def compute_embedding_mean_absolute_error(self, embedding):
        self.metrics[f'{embedding}_absolute_error_mean'] = np.mean(absolute_error(self.top_N_and_all_expected_edge_results.loc[self.top_N_and_all_expected_edge_results['expected_edge'] == True, f'predicted_{embedding}'], self.top_N_and_all_expected_edge_results.loc[self.top_N_and_all_expected_edge_results['expected_edge'] == True, f'expected_{embedding}']))
    
    def compute_embedding_mean_squared_error(self, embedding):
        self.metrics[f'{embedding}_squared_error_mean'] = np.mean(squared_error(self.top_N_and_all_expected_edge_results.loc[self.top_N_and_all_expected_edge_results['expected_edge'] == True, f'predicted_{embedding}'], self.top_N_and_all_expected_edge_results.loc[self.top_N_and_all_expected_edge_results['expected_edge'] == True, f'expected_{embedding}']))

    def get_user_positional_errors(self):
        if self.positional_error_scores is not None:
            return self.positional_error_scores
        
        filtered_rows = self.top_N_and_all_expected_edge_results.loc[self.top_N_and_all_expected_edge_results['expected_edge'] == True]
        sorted_rows_expected = filtered_rows.sort_values(by='expected_score', ascending=False)
        grouped_by_user_expected = sorted_rows_expected.groupby('user')

        positional_error_scores = {}
        for user, user_rows_expected in grouped_by_user_expected:
            if len(user_rows_expected) == 0:
                continue
            indices_expected = user_rows_expected.reset_index(drop=True).index.values
            user_predicted_rank = user_rows_expected['user_predicted_rank'].values
            sorted_indexes = {value: index for index, value in enumerate(sorted(user_predicted_rank))}
            predicted_indices = [sorted_indexes[num] for num in user_predicted_rank]
            
            positional_error_score = sum((abs(expected_index - predicted_index) for expected_index, predicted_index in zip(indices_expected, predicted_indices))) / len(predicted_indices)
            positional_error_scores[user] = positional_error_score
        self.positional_error_scores = positional_error_scores
        return positional_error_scores

    def compute_user_percentile_positional_error(self, percentile):
        positional_error_scores = self.get_user_positional_errors()
        self.metrics[f'positional_error_at_{percentile}_user_percentile'] = np.percentile(np.array([score for score in positional_error_scores.values()]), percentile)

    def plot_user_percentile_positional_error(self):
        positional_error_scores = self.get_user_positional_errors()
        self.metrics[f'positional_error_percentiles_figure'] = get_percentile_figure(list(positional_error_scores.values()), f'Positional Error Percentiles')

    def plot_log_user_percentile_positional_error(self):
        positional_error_scores = self.get_user_positional_errors()
        self.metrics[f'log_positional_error_percentiles_figure'] = get_percentile_figure(np.log(np.array(list(positional_error_scores.values())) + 1).tolist(), f'Log Positional Error Percentiles')

    # For all expected edges - compute whehter or not the ordering is correct (works for uncalibrated models). Less assumption than ROC.
    # Note: This system doesn't work for constant scoring systems.
    def compute_mean_positional_error(self):
        positional_error_scores = self.get_user_positional_errors()
        self.metrics[f'mean_positional_error'] = np.mean(np.array([score for score in positional_error_scores.values()]))

    def plot_roc_curve(self):
        roc_curve = skmetrics.roc_curve(self.top_N_edge_results['expected_edge'].astype(int).tolist(), self.top_N_edge_results['predicted_score'])
        self.metrics['roc_figure'] = get_roc_figure(roc_curve, 'User-Game Predictions ROC Curve')

    def compute_auc_roc(self):
        self.metrics['auc_roc'] = skmetrics.roc_auc_score(self.top_N_edge_results['expected_edge'].astype(int).tolist(), self.top_N_edge_results['predicted_score'])

    # This is just like the normal ROC except it orders predictions by user rank then score (so all user #1 recommendations come before any user gets their #2 recommendation). This helps prevent user predictive score bias and is more useful to real world scenarios.
    def plot_user_based_roc_curve(self):
        max_predicted_score = np.max(self.top_N_edge_results['predicted_score'])
        min_predicted_score = np.min(self.top_N_edge_results['predicted_score'])
        transformed_scores = self.top_N_edge_results['predicted_score'].apply(lambda predicted_score: linear_transformation(predicted_score, min_predicted_score, max_predicted_score, 0.99, 0.01))
        roc_curve = skmetrics.roc_curve(self.top_N_edge_results['expected_edge'].astype(int).tolist(), 1 / (self.top_N_edge_results['user_predicted_rank'] + transformed_scores))
        self.metrics['user_based_roc_figure'] = get_roc_figure(roc_curve, 'User-Game Predictions User Based ROC Curve')

    # This is just like the normal ROC except it orders predictions by user rank then score (so all user #1 recommendations come before any user gets their #2 recommendation). This helps prevent user predictive score bias and is more useful to real world scenarios.
    def compute_user_based_auc_roc(self):
        max_predicted_score = np.max(self.top_N_edge_results['predicted_score'])
        min_predicted_score = np.min(self.top_N_edge_results['predicted_score'])
        transformed_scores = self.top_N_edge_results['predicted_score'].apply(lambda predicted_score: linear_transformation(predicted_score, min_predicted_score, max_predicted_score, 0.99, 0.01))
        self.metrics['user_based_auc_roc'] = skmetrics.roc_auc_score(self.top_N_edge_results['expected_edge'].astype(int).tolist(), 1 / (self.top_N_edge_results['user_predicted_rank'] + transformed_scores))

    def save_metrics(self, folder_name, overwrite=False):
        full_folder = SAVED_EVALUATION_PATH + folder_name + '/'
        assert not os.path.exists(full_folder) or overwrite, f'Tried to save to a folder that already exists {folder_name} without allowing for overwrite.'
        if not os.path.exists(full_folder):
            os.makedirs(full_folder)
        with open(f'{full_folder}metrics.txt', 'w') as metrics_file:
            for key, value in self.metrics.items():
                if isinstance(value, float):
                    metrics_file.write(f'{key}: {value}\n')
                elif isinstance(value, plt.Figure):
                    value.savefig(f'{full_folder}{key}.png')
                    plt.close(value)