from abc import ABC, abstractmethod
from quickselect import floyd_rivest
from tqdm import tqdm
import os
import pandas as pd
from dataset.data_loader import EXTERNAL_DATA_SOURCE

SAVED_MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models/')
SAVED_NN_PATH = 'saved_nns/'
PUBLISHED_MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'published_recommendation_models/')

class BaseGameRecommendationModel(ABC):
    def __init__(self):
        schema = {'user_id': 'int64',
                  'game_id': 'int64',
                 }
        self.users_games_interactions_fine_tuned = pd.DataFrame(columns=schema.keys()).astype(schema)
    
    @abstractmethod
    def name(self):
        pass

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    # Module has gt -> ge bug. Uncomment when fixed
    # def select_scores(self, scores, N = None, should_sort=True):
    #     if N is not None and N < len(scores):
    #         nth_largest_score = floyd_rivest.nth_largest([score[1] for score in scores], N - 1)
    #         print(nth_largest_score)
    #         scores = [score for score in scores if score[1] >= nth_largest_score]
    #         print(scores)
    #     if should_sort or (N is not None and len(scores) > N):
    #         scores = sorted(scores, key=lambda x: x[1], reverse=True)
    #     if N is not None:
    #         scores = scores[:N]
    #     return scores
    
    def select_scores(self, scores, N = None, should_sort=True, games_to_filter_out=[], games_to_include=[]):
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        output_scores = sorted_scores
        if N is not None:
            output_scores = output_scores[:N + len(games_to_filter_out)]
        if len(games_to_filter_out) > 0:
            output_scores = [(game, score) for game, score in output_scores if game not in games_to_filter_out]
        if N is not None:
            output_scores = output_scores[:N]
        if len(games_to_include) > 0:
            output_games = set([game for game, score in output_scores])
            missed_games_to_include = set(games_to_include) - output_games
            output_scores = output_scores + [(game, score) for game, score in sorted_scores if game in missed_games_to_include]
        return output_scores

    # Train the model given the data loader.
    @abstractmethod
    def train(self):
        pass

    def fine_tune(self, user_id):
        all_user_games_df = self.data_loader.get_users_games_df_for_user(user_id, preprocess=True)
        all_interactions_df = self.data_loader.get_interactions_df_for_user(user_id, preprocess=True)
        # Get all then filter to do score normalization on all data but only do modifications on external data.
        external_user_games_df = all_user_games_df[all_user_games_df['source'] == EXTERNAL_DATA_SOURCE]
        external_interactions_df = all_interactions_df[all_interactions_df['source'] == EXTERNAL_DATA_SOURCE]
        def get_new_df(df):
            merged_df = pd.merge(df, self.users_games_interactions_fine_tuned, on=['user_id', 'game_id'], how='left', indicator=True)
            return merged_df[merged_df['_merge'] == 'left_only'].drop(columns='_merge')
        new_user_games_df = get_new_df(external_user_games_df)
        new_interactions_df = get_new_df(external_interactions_df)
        
        self._fine_tune(user_id, new_user_games_df, new_interactions_df, all_user_games_df, all_interactions_df)
        self.users_games_interactions_fine_tuned = pd.concat([self.users_games_interactions_fine_tuned, new_user_games_df[['user_id', 'game_id']], new_interactions_df[['user_id', 'game_id']]])

    @abstractmethod
    def _fine_tune(self, user_id, new_user_games_df, new_interactions_df, all_user_games_df, all_interactions_df):
        pass

    # TODO Phase out if not useful.
    @abstractmethod
    def get_score_between_user_and_game(self, user, game):
        pass

    # TODO Phase out if not useful.
    @abstractmethod
    def get_scores_between_users_and_games(self, users, games):
        pass

    # Output: List of top N scores (sorted) for new game recommendations for a user. Formatted as [(game_id, score)]
    @abstractmethod
    def score_and_predict_n_games_for_user(self, user, N, should_sort = True, games_to_include=[]):
        pass

    @abstractmethod
    def save(self, file_name, overwrite=False):
        pass

    @abstractmethod
    def _load(self, file_path):
        pass
    
    # NOTE: Assumes save file extension is .pkl. Abstract that if it changes.
    def model_file_exists(self, file_name):
        return os.path.exists(os.path.join(SAVED_MODELS_PATH, file_name + '.pkl'))

    def load(self, file_name, load_published_model=False):
        folder_path = PUBLISHED_MODELS_PATH if load_published_model else SAVED_MODELS_PATH
        self._load(folder_path, file_name)