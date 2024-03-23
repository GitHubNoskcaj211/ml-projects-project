from models.base_model import BaseGameRecommendationModel, SAVED_MODELS_PATH
import pickle
import random
import os

class RandomModel(BaseGameRecommendationModel):
    def __init__(self):
        super().__init__()

    def name(self):
        return 'random'

    def train(self, seed=None, seed_min=-1e9, seed_max=1e9):
        assert self.data_loader.cache_local_dataset, 'Method requires full load.'
        random.seed(seed)
        self.user_to_seed = {user_id: random.randint(seed_min, seed_max) for user_id in self.data_loader.get_user_node_ids()}
        self.game_nodes = self.data_loader.get_game_node_ids()

    def _fine_tune(self, user_id, new_user_games_df, new_interactions_df, all_user_games_df, all_interactions_df, seed=None, seed_min=-1e9, seed_max=1e9):
        random.seed(seed)
        self.user_to_seed[user_id] = random.randint(seed_min, seed_max)
    
    def get_score_between_user_and_game(self, user, game):
        # NOTE: Score between user and game will be inconsistent with score and predict n games for user. This will lead to slight inaccuracies when both are used together for example during eval.
        return random.random()

    def score_and_predict_n_games_for_user(self, user, N=None, should_sort=True):
        games_to_filter_out = self.data_loader.get_all_game_ids_for_user(user)
        random.seed(self.user_to_seed[user])
        scores = [(game, random.random()) for game in self.game_nodes if game not in games_to_filter_out]
        return self.select_scores(scores, N, should_sort)

    def save(self, file_name, overwrite=False):
        assert not os.path.isfile(SAVED_MODELS_PATH + file_name + '.pkl') or overwrite, f'Tried to save to a file that already exists {file_name} without allowing for overwrite.'
        with open(SAVED_MODELS_PATH + file_name + '.pkl', 'wb') as file:
            pickle.dump({
                'user_to_seed': self.user_to_seed,
                'game_nodes': self.game_nodes,
            }, file)

    def _load(self, folder_path, file_name):
        with open(folder_path + file_name + '.pkl', 'rb') as file:
            loaded_obj = pickle.load(file)
            self.user_to_seed = loaded_obj['user_to_seed']
            self.game_nodes = loaded_obj['game_nodes']