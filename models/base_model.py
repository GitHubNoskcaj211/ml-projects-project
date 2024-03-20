from abc import ABC, abstractmethod
from quickselect import floyd_rivest
from tqdm import tqdm
import os
import pandas as pd

SAVED_MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models/')
SAVED_NN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_nns/')
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
    
    def select_scores(self, scores, N = None, should_sort=True):
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        if N is not None:
            scores = scores[:N]
        return scores

    # Train the model given the data loader.
    @abstractmethod
    def train(self):
        pass

    def fine_tune(self, user_id):
        user_games_df = self.data_loader.get_users_games_df_for_user(user_id, get_local=False)
        interactions_df = self.data_loader.get_interactions_df_for_user(user_id, get_local=False)
        def get_new_df(df):
            merged_df = pd.merge(df, self.users_games_interactions_fine_tuned, on=['user_id', 'game_id'], how='left', indicator=True)
            return merged_df[merged_df['_merge'] == 'left_only'].drop(columns='_merge')
        new_user_games_df = get_new_df(user_games_df)
        new_interactions_df = get_new_df(interactions_df)

        self._fine_tune(user_id, new_user_games_df, new_interactions_df)
        self.users_games_interactions_fine_tuned = pd.concat([self.users_games_interactions_fine_tuned, new_user_games_df[['user_id', 'game_id']], new_interactions_df[['user_id', 'game_id']]])

    @abstractmethod
    def _fine_tune(self, user_id, new_user_games_df, new_interactions_df):
        pass

    @abstractmethod
    def get_score_between_user_and_game(self, user, game):
        pass

    # Input: 
    # Output: list of nodes 
    def recommend_n_games_for_user(self, user, N=None, should_sort=True):
        scores = self.score_and_predict_n_games_for_user(user, N, should_sort=should_sort)
        return [game for game, score in scores]

    # def predict_for_all_users(self, N):
    #     all_predictions_and_scores_per_user = {}
    #     pool = multiprocessing.Pool(processes=16, maxtasksperchild=1)
    #     for node, data in self.data_loader.test_network.nodes(data=True):
    #         if data['node_type'] != NodeType.USER:
    #             continue
    #         all_predictions_and_scores_per_user[node] = pool.apply_async(self.score_and_predict_n_games_for_user, args=(node, N))
    #     pool.close()
    #     pool.join()
    #     for node, async_result in all_predictions_and_scores_per_user.items():
    #         all_predictions_and_scores_per_user[node] = async_result.get()
    #     return all_predictions_and_scores_per_user

    def predict_for_all_users(self, N, should_sort=True):
        user_nodes = self.data_loader.users_df['id'].to_list()
        all_predictions_and_scores_per_user = dict.fromkeys(user_nodes)
        for node in tqdm(user_nodes, desc='User Predictions'):
            all_predictions_and_scores_per_user[node] = self.score_and_predict_n_games_for_user(node, N=N, should_sort=should_sort)
        return all_predictions_and_scores_per_user

    # Output: List of top N scores (sorted) for new game recommendations for a user. Formatted as [(game_id, score)]
    @abstractmethod
    def score_and_predict_n_games_for_user(self, user, N, should_sort = True):
        pass

    @abstractmethod
    def save(self, file_name, overwrite=False):
        pass

    @abstractmethod
    def _load(self, file_path):
        pass

    def load(self, file_name, load_published_model=False):
        folder_path = PUBLISHED_MODELS_PATH if load_published_model else SAVED_MODELS_PATH
        self._load(folder_path + file_name)