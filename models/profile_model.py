import sys
import os
sys.path.append(os.path.abspath(''))

import tracemalloc
import time
import argparse

tracemalloc.start()
start_time = time.perf_counter()

print("Importing")
from dataset.data_loader import DataLoader

from models.collaborative_filtering_model import CollaborativeFiltering  # noqa: F401
from models.common_neighbors_model import CommonNeighborsModelStoragePredictEfficient, CommonNeighborsModelLoadPredictEfficient, CommonNeighborsModelStorageMemoryEfficient   # noqa: F401
from models.ncf_model import NCFModel    # noqa: F401
from models.popularity_model import GamePopularityModel    # noqa: F401

model_dispatcher = {
    'common_neighbors_storage_predict_efficient': CommonNeighborsModelStoragePredictEfficient,
    'common_neighbors_load_predict_efficient': CommonNeighborsModelLoadPredictEfficient,
    'common_neighbors_storage_memory_efficient': CommonNeighborsModelStorageMemoryEfficient,
    'game_popularity': GamePopularityModel,
}
 
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help = "Model name to use", choices=list(model_dispatcher.keys()), required=True)
parser.add_argument("-f", "--load_file_name", help = "Model save to load in", type=str, required=True)
parser.add_argument("-N", "--num_games_to_recommend", help = "Num games to recommend", type=int, default=50)
args = parser.parse_args()
test_user_id = 76561198835352289

import tracemalloc
import time

model = model_dispatcher[args.model]()
file_name = args.load_file_name
N = args.num_games_to_recommend

try:
    print("Initializing Data Loader")
    data_loader = DataLoader()
    model.set_data_loader(data_loader)
    print('Loading Model')
    model.load(file_name)
except NotImplementedError:
    print("Initializing Data Loader")
    data_loader = DataLoader(full_load=True)
    model.set_data_loader(data_loader)
    network = data_loader.get_full_network()
    print('Training')
    train_network, test_network = data_loader.load_stratified_user_train_test_network(network=network, train_percentage=0.8, test_percentage=0.2, seed=0)
    model.train(train_network)
print('Recommending')
preds = model.recommend_n_games_for_user(test_user_id, N)

end_time = time.perf_counter()
size, peak = tracemalloc.get_traced_memory()


def get_human_readable(num):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num < 1000:
            return f"{num:.3f} {unit}"
        num /= 1000
    assert False


print()
print("Current Memory Usage:", get_human_readable(size))
print("Peak Memory Usage:", get_human_readable(peak))
print("Elapsed: ", end_time - start_time, "seconds")
tracemalloc.stop()
