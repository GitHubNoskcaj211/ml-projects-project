from evaluation.evaluation_harness_harness import Evaluator
import cProfile

import sys
sys.path.append("../dataset")
sys.path.append("../models")
from dataset.data_loader import DataLoader
from models.random_model import RandomModel
from models.common_neighbors_model import CommonNeighborsModel
from models.popularity_model import GamePopularityModel

data_loader = DataLoader()
network = data_loader.get_full_network()
train_network, test_network = data_loader.load_stratified_user_train_test_network(network=network, train_percentage=0.9, test_percentage=0.1, seed=0)
evaluator = Evaluator(test_network, top_N_games_to_eval=500)

model = CommonNeighborsModel()
model.set_data_loader(data_loader)
model.train(train_network)

def profile_evaluator():
    evaluator.reset(model)
    evaluator.compute_top_N_hit_percentage_at_user_percentile(10, 75)

cProfile.run('profile_evaluator()')