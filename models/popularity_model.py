from models.base_model import BaseGameRecommendationModel, SAVED_MODELS_PATH
import pickle
import os

# Recommend in order of most to least popular games (based on number of edges).
class GamePopularityModel(BaseGameRecommendationModel):
    def __init__(self):
        super().__init__()

    def name(self):
        return 'game_popularity'

    def train(self):
        assert self.data_loader.cache_local_dataset, 'Method requires full load.'
        game_nodes = self.data_loader.get_game_node_ids()
        train_users_games_df = self.data_loader.users_games_df[self.data_loader.users_games_df['data_split'] == 'train']
        user_degree_counts = train_users_games_df.groupby('game_id').size().reset_index(name='user_degree')
        degrees = {row['game_id']: row['user_degree'] for index, row in user_degree_counts.iterrows()}
        degrees = {game: degrees[game] if game in degrees else 0 for game in game_nodes}
        score_fn = lambda game: degrees[game]
        self.scores = [(game, score_fn(game)) for game in game_nodes]
        self.scores = sorted(self.scores, key=lambda x: x[1], reverse=True)
        self.game_to_score_index = {game: ii for ii, (game, _) in enumerate(self.scores)}

    def get_score_between_user_and_game(self, user, game):
        return self.scores[self.game_to_score_index[game]][1]
    
    def get_scores_between_users_and_games(self, users, games):
        assert len(users) == len(games), 'Inconsistent list lengths.'
        return [self.scores[self.game_to_score_index[game]][1] for game in games]
    
    def _fine_tune(self, user_id, new_user_games_df, new_interactions_df, all_user_games_df, all_interactions_df):
        pass
        # TODO

    def score_and_predict_n_games_for_user(self, user, N=None, should_sort=True, games_to_include=[]):
        games_to_filter_out = self.data_loader.get_all_game_ids_for_user(user)
        return self.select_scores(self.scores, N, should_sort, games_to_filter_out=games_to_filter_out, games_to_include=games_to_include)

    def save(self, file_name, overwrite=False):
        assert not os.path.isfile(SAVED_MODELS_PATH + file_name + '.pkl') or overwrite, f'Tried to save to a file that already exists {file_name} without allowing for overwrite.'
        with open(SAVED_MODELS_PATH + file_name + '.pkl', 'wb') as file:
            pickle.dump({
                'scores': self.scores,
                'game_to_score_index': self.game_to_score_index,
            }, file)

    def _load(self, folder_path, file_name):
        with open(folder_path + file_name + '.pkl', 'rb') as file:
            loaded_obj = pickle.load(file)
            self.scores = loaded_obj['scores']
            self.game_to_score_index = loaded_obj['game_to_score_index']