from abc import ABC, abstractmethod

class BaseGameRecommendationModel(ABC):
    @abstractmethod
    def name(self):
        pass

    # Train the model given the data loader.
    @abstractmethod
    def train(self, data_loader):
        pass

    # Input: 
    # Output: list of nodes 
    @abstractmethod
    def recommend_n_games_for_user(self, user, N):
        pass

    @abstractmethod
    def save(self, file_name, overwrite=False):
        pass

    @abstractmethod
    def load(self, file_name):
        pass