from models.base_model import BaseGameRecommendationModel
from dataset.data_loader import NodeType

# Recommend in order of most to least popular games (based on number of edges).
class GamePopularityModel(BaseGameRecommendationModel):
    def __init__(self):
        pass

    def name(self):
        return 'game_popularity'

    def train(self):
        self.game_nodes = [node for node, data in self.data_loader.train_network.nodes(data=True) if data['node_type'] == NodeType.GAME]
        self.degrees = {node: val for (node, val) in self.data_loader.train_network.degree()}
        score_fn = lambda game: self.degrees[game]
        self.scores = [(game, score_fn(game)) for game in self.game_nodes]
        self.scores = sorted(self.scores, key=lambda x: x[1], reverse=True)
        self.game_to_score_index = {game: ii for ii, (game, _) in enumerate(self.scores)}

    def get_score_between_user_and_game(self, user, game):
        return self.scores[self.game_to_score_index[game]][1]

    def score_and_predict_n_games_for_user(self, user, N=None, should_sort=True):
        root_node_neighbors = list(self.data_loader.train_network.neighbors(user))
        scores_for_user = [(game, embeddings) for game, embeddings in self.scores if game not in root_node_neighbors]
        if N is not None:
            scores_for_user = scores_for_user[:N]
        return scores_for_user

    def save(self, file_name, overwrite=False):
        raise NotImplementedError('Did not implement saving on popularity model.')

    def load(self, file_name):
        raise NotImplementedError('Did not implement loading on popularity model.')
