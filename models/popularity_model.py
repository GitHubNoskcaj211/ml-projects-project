from models.base_model import BaseGameRecommendationModel, SAVED_MODELS_PATH
from dataset.data_loader import NodeType
import pickle
import os

# Recommend in order of most to least popular games (based on number of edges).
class GamePopularityModel(BaseGameRecommendationModel):
    def __init__(self):
        pass

    def name(self):
        return 'game_popularity'

    def train(self):
        game_nodes = self.data_loader.get_game_node_ids()
        train_users_games_df = self.data_loader.users_games_df[self.data_loader.users_games_df['train_split']]
        user_degree_counts = train_users_games_df.groupby('game_id').size().reset_index(name='user_degree')
        degrees = {row['game_id']: row['user_degree'] for index, row in user_degree_counts.iterrows()}
        degrees = {game: degrees[game] if game in degrees else 0 for game in game_nodes}
        score_fn = lambda game: degrees[game]
        self.scores = [(game, score_fn(game)) for game in game_nodes]
        self.scores = sorted(self.scores, key=lambda x: x[1], reverse=True)
        self.game_to_score_index = {game: ii for ii, (game, _) in enumerate(self.scores)}

    def get_score_between_user_and_game(self, user, game):
        return self.scores[self.game_to_score_index[game]][1]

    def score_and_predict_n_games_for_user(self, user, N=None, should_sort=True):
        user_games_df = self.data_loader.get_users_games_df_for_user(user)
        games_to_filter_out = user_games_df['game_id'].to_list()
        scores_for_user = [(game, embeddings) for game, embeddings in self.scores if game not in games_to_filter_out]
        if N is not None:
            scores_for_user = scores_for_user[:N]
        return scores_for_user

    def save(self, file_name, overwrite=False):
        assert not os.path.isfile(SAVED_MODELS_PATH + file_name + '.pkl') or overwrite, f'Tried to save to a file that already exists {file_name} without allowing for overwrite.'
        with open(SAVED_MODELS_PATH + file_name + '.pkl', 'wb') as file:
            pickle.dump({
                'scores': self.scores,
                'game_to_score_index': self.game_to_score_index,
            }, file)

    def _load(self, file_path):
        with open(file_path + '.pkl', 'rb') as file:
            loaded_obj = pickle.load(file)
            self.scores = loaded_obj['scores']
            self.game_to_score_index = loaded_obj['game_to_score_index']