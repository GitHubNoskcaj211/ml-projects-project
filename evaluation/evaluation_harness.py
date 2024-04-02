import os
import numpy as np
import pandas as pd
from sklearn import metrics as skmetrics
from matplotlib import pyplot as plt
from utils.utils import linear_transformation
from abc import ABC, abstractmethod

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


# TODO Add interactions to this.
class Evaluator(ABC):
    def __init__(self, data_loader, top_N_games_to_eval, num_users_to_eval=None, seed=0, debug=False):
        self.data_loader = data_loader
        assert self.data_loader.cache_local_dataset, 'Evaluator requires data fully loaded.'
        self.top_N_games_to_eval = top_N_games_to_eval
        self.num_users_to_eval = num_users_to_eval
        if num_users_to_eval is not None:
            self.users_to_eval = self.data_loader.users_df['id'].sample(n=num_users_to_eval, random_state=seed)
        else:
            self.users_to_eval = self.data_loader.users_df['id']
        self.debug = debug

    def create_load_prepare_save_model(self, network_initializer, network_save_file):
        self.model = network_initializer()
        self.model.set_data_loader(self.data_loader)
        if self.model.model_file_exists(network_save_file):
            if self.debug:
                print('Loading model:', network_save_file)
            self.model.load(network_save_file)
            if self.debug:
                print('Doen loading model.', network_save_file)
        else:
            if self.debug:
                print('Preparing model.')
            self.prepare_model()
            self.model.save(network_save_file)
            if self.debug:
                print('Done preparing model.')

    @abstractmethod
    def prepare_model(self, model):
        pass

    def reset(self, network_initializer, network_save_file):
        self.create_load_prepare_save_model(network_initializer, network_save_file)
        self.metrics = {}
        self.roc_curve = None
        self.positional_error_scores = None

        if self.debug:
            print('Getting predictions.')
        self.top_N_edge_results = self.model.predict_for_all_users(N = self.top_N_games_to_eval, users_to_predict=self.users_to_eval)
        if self.debug:
            print('Done getting predictions.')

        if self.debug:
            print('Appending dataframe information.')
        merged_df = pd.merge(self.top_N_edge_results, self.data_loader.users_games_df, 
                            left_on=['user', 'game'], right_on=['user_id', 'game_id'], 
                            how='left', suffixes=('', '_y'))
        merged_df['expected_edge'] = ~merged_df['user_id'].isnull()
        merged_df['expected_score'] = merged_df['score'] if 'score' in merged_df.columns else None
        assert merged_df.loc[merged_df['expected_edge'], 'data_split'].eq('test').all()
        merged_df.drop(columns=self.data_loader.users_games_df.columns, inplace=True, errors='ignore')
        self.top_N_edge_results = merged_df
        if self.debug:
            print('Ranking top N.')
        self.top_N_edge_results['user_predicted_rank'] = self.top_N_edge_results.groupby('user')['predicted_score'].rank(ascending=False) - 1

        if self.debug:
            print('Constructing missed edge dataframe.')
        test_data = self.data_loader.users_games_df[(self.data_loader.users_games_df['data_split'] == 'test') & (self.data_loader.users_games_df['user_id'].isin(self.users_to_eval))]
        merged_data = pd.merge(self.top_N_edge_results, test_data, left_on=['user', 'game'], right_on=['user_id', 'game_id'], how='right', indicator=True)
        missed_test_edges = merged_data[merged_data['_merge'] == 'right_only']
        expected_missed_edge_data = []
        for index, row in missed_test_edges.iterrows():  # Iterate over rows, not just columns
            expected_missed_edge_data.append({'user': row['user_id'], 'game': row['game_id'], 'expected_edge': True, 'predicted_score': self.model.get_score_between_user_and_game(row['user_id'], row['game_id']), 'expected_score': row['score']})
        missed_expected_edge_results = pd.DataFrame(expected_missed_edge_data)
        if self.debug:
            print('Ranking missed.')
        missed_expected_edge_results['user_predicted_rank'] = missed_expected_edge_results.groupby('user')['predicted_score'].rank(ascending=False) - 1 + self.top_N_games_to_eval
        self.top_N_and_all_expected_edge_results = pd.concat([self.top_N_edge_results, missed_expected_edge_results], ignore_index=True)

        if self.debug:
            print('Done getting edge results.')

    def compute_top_N_recall(self, N):
        assert N < self.top_N_games_to_eval, 'Cannot get top N recall when we have less top N games to eval since the dataframe is malformed.'
        top_N_rows = self.top_N_and_all_expected_edge_results[(self.top_N_and_all_expected_edge_results['user_predicted_rank'] < N)]
        num_top_N_recalls = (top_N_rows['expected_edge'] == True).sum()
        num_expected_edges = (self.top_N_and_all_expected_edge_results['expected_edge'] == True).sum()
        self.metrics[f'top_{N}_recall'] = 1.0 if num_expected_edges == 0 else num_top_N_recalls / num_expected_edges

    def get_top_N_recall_per_user(self, N):
        assert N < self.top_N_games_to_eval, 'Cannot get top N recall when we have less top N games to eval since the dataframe is malformed.'
        filtered_rows = self.top_N_and_all_expected_edge_results[(self.top_N_and_all_expected_edge_results['user_predicted_rank'] < N)]
        user_top_N_recalls = filtered_rows.groupby('user')['expected_edge'].sum().reset_index(name='tps')
        expected_edges_per_user = self.data_loader.users_games_df[self.data_loader.users_games_df['data_split'] == 'test'].groupby('user_id').size()
        user_top_N_recalls['num_expected_games'] = user_top_N_recalls['user'].map(expected_edges_per_user)
        user_top_N_recalls['recall'] = user_top_N_recalls['tps'] / user_top_N_recalls['num_expected_games']
        return user_top_N_recalls

    def plot_top_N_recall_percentiles(self, N):
        user_top_N_recalls = self.get_top_N_recall_per_user(N)
        self.metrics[f'top_{N}_recall_user_percentiles_figure'] = get_percentile_figure(user_top_N_recalls.loc[user_top_N_recalls['num_expected_games'] != 0, 'recall'].values, f'User Top {N} Recall Percentage Percentiles')
    
    # Percentile is an integer 0-100.
    def compute_top_N_recall_at_user_percentile(self, N, percentile):
        user_top_N_recalls = self.get_top_N_recall_per_user(N)
        self.metrics[f'top_{N}_recall_at_{percentile}_user_percentile'] = np.percentile(user_top_N_recalls.loc[user_top_N_recalls['num_expected_games'] != 0, 'recall'], percentile)

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
    
class TrainEvaluator(Evaluator):
    def __init__(self, data_loader, top_N_games_to_eval, num_users_to_eval=None, seed=0, debug=False):
        super().__init__(data_loader, top_N_games_to_eval, num_users_to_eval, seed, debug)
    
    def prepare_model(self):
        self.model.train()
