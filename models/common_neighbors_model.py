from base_model import BaseGameRecommendationModel

class CommonNeighbors(BaseGameRecommendationModel):
    def __init__(self):
        pass

    def name(self):
        return 'common_neighbors'

    def train(self, data_loader):
        pass

    def recommend_n_games_for_user(self, user, N):
        pass

    def save(self, file_name, overwrite=False):
        pass

    def load(self, file_name):
        pass