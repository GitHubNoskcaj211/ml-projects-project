from abc import ABC, abstractmethod
import numpy as np

SAVED_MODELS_PATH = 'saved_models/'

class BaseGameRecommendationModel(ABC):
    @abstractmethod
    def name(self):
        pass

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    # Train the model given the data loader.
    @abstractmethod
    def train(self):
        pass

    # Input: 
    # Output: list of nodes 
    def recommend_n_games_for_user(self, user, N=None):
        scores = self.score_n_games_for_user(user, N)
        assert np.all(np.diff(np.array([score for game, score, embedding_predictions in scores])) <= 0), "Score array is not sorted in reverse order."
        return [game for game, score, embedding_predictions in scores]

    # Output: List of top N scores (sorted) for new game recommendations for a user. Formatted as [(game_id, score, embedding_predictions)]
    @abstractmethod
    def score_and_predict_n_games_for_user(self, user, N):
        pass

    @abstractmethod
    def save(self, file_name, overwrite=False):
        pass

    @abstractmethod
    def load(self, file_name):
        pass