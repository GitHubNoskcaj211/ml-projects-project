import os
import numpy as np
import pandas as pd
from sklearn import metrics as skmetrics
from matplotlib import pyplot as plt
from utils.utils import linear_transformation
from abc import ABC, abstractmethod
import itertools
import pandas as pd
from utils.firestore import DatabaseClient
from google.cloud.firestore_v1.base_query import FieldFilter
from tqdm import tqdm

from compare_auc_delong_xu import delong_roc_variance

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
    if len(sorted_values) - 1 == 0:
        print('Not enough samples to get percentile figure.')
        return
    percentiles = [i / (len(sorted_values) - 1) for i in range(len(sorted_values))]
    axis.plot(percentiles, sorted_values)
    axis.set_title(metric_title)
    axis.set_xlabel('User Percentiles')
    axis.set_ylabel('Metric Value')
    return fig


# TODO Add interactions to this.
class Evaluator(ABC):
    def __init__():
        pass

    @abstractmethod
    def reset(self):
        self.metrics = {}
        self.roc_curve = None
        self.positional_error_scores = None
        self.name = ""
    
    def compute_top_N_hit_percentage(self, N):
        assert self.top_N_games_to_eval is None or N < self.top_N_games_to_eval, 'Cannot get top N hit_percentage when we have less top N games to eval since the dataframe is malformed.'
        top_N_rows = self.results_df[(self.results_df['user_predicted_rank'] < N)]
        num_top_N_hit_percentages = (top_N_rows['expected_edge'] == True).sum()
        self.metrics[f'top_{N}_hit_percentage'] = 1.0 if len(top_N_rows) == 0 else num_top_N_hit_percentages / len(top_N_rows)

    def get_top_N_hit_percentage_per_user(self, N):
        assert self.top_N_games_to_eval is None or N < self.top_N_games_to_eval, 'Cannot get top N hit_percentage when we have less top N games to eval since the dataframe is malformed.'
        filtered_rows = self.results_df[(self.results_df['user_predicted_rank'] < N)]
        user_top_N_hit_percentages = filtered_rows.groupby('user')['expected_edge'].sum().reset_index(name='tps')
        user_top_N_hit_percentages['num_expected_games'] = user_top_N_hit_percentages['user'].map(filtered_rows.groupby('user').size())
        user_top_N_hit_percentages['num_expected_games'] = user_top_N_hit_percentages['num_expected_games'].fillna(0)
        user_top_N_hit_percentages['hit_percentage'] = user_top_N_hit_percentages['tps'] / user_top_N_hit_percentages['num_expected_games']
        return user_top_N_hit_percentages

    def plot_top_N_hit_percentage_percentiles(self, N):
        user_top_N_hit_percentages = self.get_top_N_hit_percentage_per_user(N)
        self.metrics[f'top_{N}_hit_percentage_user_percentiles_figure'] = get_percentile_figure(user_top_N_hit_percentages.loc[user_top_N_hit_percentages['num_expected_games'] != 0, 'hit_percentage'].values, f'User Top {N} Hit Percentage Percentiles')
    
    # Percentile is an integer 0-100.
    def compute_top_N_hit_percentage_at_user_percentile(self, N, percentile):
        user_top_N_hit_percentages = self.get_top_N_hit_percentage_per_user(N)
        self.metrics[f'top_{N}_hit_percentage_at_{percentile}_user_percentile'] = np.percentile(user_top_N_hit_percentages.loc[user_top_N_hit_percentages['num_expected_games'] != 0, 'hit_percentage'], percentile)

    def compute_top_N_recall(self, N):
        assert self.top_N_games_to_eval is None or N < self.top_N_games_to_eval, 'Cannot get top N recall when we have less top N games to eval since the dataframe is malformed.'
        top_N_rows = self.results_df[(self.results_df['user_predicted_rank'] < N)]
        num_top_N_recalls = (top_N_rows['expected_edge'] == True).sum()
        num_expected_edges = (self.results_df['expected_edge'] == True).sum()
        self.metrics[f'top_{N}_recall'] = 1.0 if num_expected_edges == 0 else num_top_N_recalls / num_expected_edges

    def get_top_N_recall_per_user(self, N):
        assert self.top_N_games_to_eval is None or N < self.top_N_games_to_eval, 'Cannot get top N recall when we have less top N games to eval since the dataframe is malformed.'
        filtered_rows = self.results_df[(self.results_df['user_predicted_rank'] < N)]
        user_top_N_recalls = filtered_rows.groupby('user')['expected_edge'].sum().reset_index(name='tps')
        expected_edges_per_user = self.results_df[self.results_df['expected_edge'] == True].groupby('user').size()
        user_top_N_recalls['num_expected_games'] = user_top_N_recalls['user'].map(expected_edges_per_user)
        user_top_N_recalls['num_expected_games'] = user_top_N_recalls['num_expected_games'].fillna(0)
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
        self.metrics[f'{embedding}_absolute_error_at_percentile_{percentile}'] = np.percentile(absolute_error(self.results_df.loc[self.results_df['expected_edge'] == True, f'predicted_{embedding}'], self.results_df.loc[self.results_df['expected_edge'] == True, f'expected_{embedding}']), percentile)
    
    # Percentile is an integer 0-100.
    def compute_embedding_percentile_squared_error(self, embedding, percentile):
        self.metrics[f'{embedding}_squared_error_at_percentile_{percentile}'] = np.percentile(squared_error(self.results_df.loc[self.results_df['expected_edge'] == True, f'predicted_{embedding}'], self.results_df.loc[self.results_df['expected_edge'] == True, f'expected_{embedding}']), percentile)
    
    def plot_embedding_percentile_absolute_error(self, embedding):
        self.metrics[f'{embedding}_absolute_error_percentiles_figure'] = get_percentile_figure(absolute_error(self.results_df.loc[self.results_df['expected_edge'] == True, f'predicted_{embedding}'], self.results_df.loc[self.results_df['expected_edge'] == True, f'expected_{embedding}']).values, f'Embedding {embedding} Absolute Error at Percentiles')
    
    def plot_embedding_percentile_squared_error(self, embedding):
        self.metrics[f'{embedding}_squared_error_percentiles_figure'] = get_percentile_figure(squared_error(self.results_df.loc[self.results_df['expected_edge'] == True, f'predicted_{embedding}'], self.results_df.loc[self.results_df['expected_edge'] == True, f'expected_{embedding}']).values, f'Embedding {embedding} Squared Error at Percentiles')

    def compute_embedding_mean_absolute_error(self, embedding):
        self.metrics[f'{embedding}_absolute_error_mean'] = np.mean(absolute_error(self.results_df.loc[self.results_df['expected_edge'] == True, f'predicted_{embedding}'], self.results_df.loc[self.results_df['expected_edge'] == True, f'expected_{embedding}']))
    
    def compute_embedding_mean_squared_error(self, embedding):
        self.metrics[f'{embedding}_squared_error_mean'] = np.mean(squared_error(self.results_df.loc[self.results_df['expected_edge'] == True, f'predicted_{embedding}'], self.results_df.loc[self.results_df['expected_edge'] == True, f'expected_{embedding}']))

    def get_user_positional_errors(self):
        if self.positional_error_scores is not None:
            return self.positional_error_scores
        
        filtered_rows = self.results_df.loc[self.results_df['expected_edge'] == True]
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
        roc_curve = skmetrics.roc_curve(self.top_N_results_df['expected_edge'].astype(int).tolist(), self.top_N_results_df['predicted_score'])
        self.metrics['roc_figure'] = get_roc_figure(roc_curve, 'User-Game Predictions ROC Curve')

    def compute_auc_roc(self):
        self.metrics['auc_roc'] = skmetrics.roc_auc_score(self.top_N_results_df['expected_edge'].astype(int).tolist(), self.top_N_results_df['predicted_score'])

    # This is just like the normal ROC except it orders predictions by user rank then score (so all user #1 recommendations come before any user gets their #2 recommendation). This helps prevent user predictive score bias and is more useful to real world scenarios.
    def plot_user_rank_roc_curve(self):
        
        roc_curve = skmetrics.roc_curve(self.top_N_results_df['expected_edge'].astype(int).tolist(), 1 / (self.top_N_results_df['user_predicted_rank'] + 1))
        self.metrics['user_rank_roc_figure'] = get_roc_figure(roc_curve, f'User-Game Predictions User Rank ROC Curve for {self.name}')

    # This is just like the normal ROC except it orders predictions by user rank then score (so all user #1 recommendations come before any user gets their #2 recommendation). This helps prevent user predictive score bias and is more useful to real world scenarios.
    def compute_user_rank_auc_roc(self):
        true_df = self.top_N_results_df['expected_edge'].astype(int)
        if np.all(true_df == 0):
            print("Not enough samples for AUC ROC")
            self.metrics["user_rank_auc_roc"] = None
            self.metrics["user_rank_auc_roc_variance"] = None
        else:
            self.metrics['user_rank_auc_roc'], self.metrics['user_rank_auc_roc_variance'] = delong_roc_variance(true_df, 1 / (self.top_N_results_df['user_predicted_rank'] + 1))

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

def include_coldstart(interaction):
    return interaction["num_game_interactions_local"] == 0 or interaction["num_game_owned_local"] == 0

def include_all(interaction):
    return True

def score_time_spent(interaction):
    return interaction["time_spent"]

def score_constant(interaction):
    return 1

class OnlineEvaluator(Evaluator):
    def __init__(self, include_fn, score_fn):
        self.include_fn = include_fn
        self.score_fn = score_fn
        self.client = DatabaseClient()
        self.top_N_games_to_eval = None
        self.all_results = self.get_results()

    def reset(self, rec_model_name, rec_model_save_path):
        super().reset()
        self.name = f"{rec_model_name} {rec_model_save_path}"
        self.results_df = self.all_results[(self.all_results['rec_model_name'] == rec_model_name) & (self.all_results['rec_model_save_path'] == rec_model_save_path)]
        self.top_N_results_df = self.results_df
    
    def get_filtered_interactions(self, include_fn):
        users = self.client.interactions_ref.document("data").collections()

        def get_interactions_for_user(user_collection):
            query = user_collection.where(filter=FieldFilter("time_spent", ">", 0.2))
            return (interaction.to_dict() for interaction in query.stream())
        interactions = (get_interactions_for_user(user_collection) for user_collection in users)
        return filter(lambda x: include_fn(x), itertools.chain.from_iterable(interactions))

    def get_results(self):
        data = pd.DataFrame(self.get_filtered_interactions(self.include_fn))
        data["expected_score"] = data.apply(self.score_fn, axis=1)
        data.rename(columns={
            "user_liked": "expected_edge",
            "user_id": "user",
            "game_id": "game",
        }, inplace=True)
        data["user_predicted_rank"] = data.groupby(['user', 'rec_model_save_path'])["timestamp"].rank()

        return data

class OfflineEvaluator(Evaluator):
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
        super().reset()
        self.create_load_prepare_save_model(network_initializer, network_save_file)

        if self.debug:
            print('Getting predictions.')

        user_ids = []
        game_ids = []
        predicted_scores = []
        for user in tqdm(self.users_to_eval, desc='User Predictions'):
            users_games_df = self.data_loader.users_games_df_grouped_by_user.get_group(user)
            predictions = self.model.score_and_predict_n_games_for_user(user, N=self.top_N_games_to_eval, should_sort=False, games_to_include=users_games_df[users_games_df['data_split'] == 'test']['game_id'].tolist())
            user_ids.extend([user] * len(predictions))
            game_ids.extend([game_id for game_id, _ in predictions])
            predicted_scores.extend([score for _, score in predictions])
        self.results_df = pd.DataFrame({
            'user': user_ids,
            'game': game_ids,
            'predicted_score': predicted_scores
        })

        if self.debug:
            print('Done getting predictions.')

        if self.debug:
            print('Appending dataframe information.')
        merged_df = pd.merge(self.results_df, self.data_loader.users_games_df, 
                            left_on=['user', 'game'], right_on=['user_id', 'game_id'], 
                            how='left', suffixes=('', '_y'))
        merged_df['expected_edge'] = ~merged_df['user_id'].isnull()
        merged_df['expected_score'] = merged_df['score'] if 'score' in merged_df.columns else None
        assert merged_df.loc[merged_df['expected_edge'], 'data_split'].eq('test').all()
        merged_df.drop(columns=self.data_loader.users_games_df.columns, inplace=True, errors='ignore')
        self.results_df = merged_df
        if self.debug:
            print('Ranking top N.')
        self.results_df['user_predicted_rank'] = self.results_df.groupby('user')['predicted_score'].rank(ascending=False) - 1
        self.top_N_results_df = self.results_df[self.results_df['user_predicted_rank'] < self.top_N_games_to_eval]

        if self.debug:
            print('Done getting edge results.')

class TrainEvaluator(OfflineEvaluator):
    def __init__(self, data_loader, top_N_games_to_eval, num_users_to_eval=None, seed=0, debug=False):
        super().__init__(data_loader, top_N_games_to_eval, num_users_to_eval, seed, debug)
    
    def prepare_model(self):
        self.model.train()

class WarmFineTuneEvaluator(OfflineEvaluator):
    def __init__(self, data_loader, top_N_games_to_eval, num_users_to_eval=None, seed=0, debug=False, fine_tune_batch_size=10):
        super().__init__(data_loader, top_N_games_to_eval, num_users_to_eval, seed, debug)
        self.fine_tune_batch_size = fine_tune_batch_size
    
    def prepare_model(self):
        self.model.train()
        # TODO Add interactions.
        for user in tqdm(self.users_to_eval, desc='Fine tuning'):
            users_games_df = self.data_loader.users_games_df_grouped_by_user.get_group(user)
            all_users_games_df = users_games_df[users_games_df['data_split'] != 'test']
            fake_interactions = pd.DataFrame(columns=users_games_df.columns)
            new_users_games_df = users_games_df[users_games_df['data_split'] == 'tune']
            new_users_games_df_shuffled = new_users_games_df.sample(frac=1).reset_index(drop=True)
            for i in range(0, len(new_users_games_df_shuffled), self.fine_tune_batch_size):
                users_games_batch_size = new_users_games_df_shuffled.iloc[i:i+self.fine_tune_batch_size]
                self.model._fine_tune(user, users_games_batch_size, fake_interactions, all_users_games_df, fake_interactions)

class ColdFineTuneEvaluator(OfflineEvaluator):
    def __init__(self, data_loader, top_N_games_to_eval, num_users_to_eval=None, seed=0, debug=False, fine_tune_batch_size=10):
        super().__init__(data_loader, top_N_games_to_eval, num_users_to_eval, seed, debug)
        self.fine_tune_batch_size = fine_tune_batch_size
    
    def prepare_model(self):
        train_df = self.data_loader.users_games_df[self.data_loader.users_games_df['data_split'] == 'train']
        train_user_ids = train_df['user_id'].unique().tolist()
        self.model.train(user_node_ids=train_user_ids)
        # TODO Add interactions.
        for user in tqdm(self.users_to_eval, desc='Fine tuning'):
            users_games_df = self.data_loader.users_games_df_grouped_by_user.get_group(user)
            all_users_games_df = users_games_df[users_games_df['data_split'] != 'test']
            fake_interactions = pd.DataFrame(columns=users_games_df.columns)
            new_users_games_df = users_games_df[users_games_df['data_split'] == 'tune']
            new_users_games_df_shuffled = new_users_games_df.sample(frac=1).reset_index(drop=True)
            if len(new_users_games_df_shuffled) == 0:
                self.model._fine_tune(user, new_users_games_df_shuffled, fake_interactions, all_users_games_df, fake_interactions)
                continue
            for i in range(0, len(new_users_games_df_shuffled), self.fine_tune_batch_size):
                users_games_batch_size = new_users_games_df_shuffled.iloc[i:i+self.fine_tune_batch_size]
                self.model._fine_tune(user, users_games_batch_size, fake_interactions, all_users_games_df, fake_interactions)
