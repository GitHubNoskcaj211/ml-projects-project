import tracemalloc
import time

tracemalloc.start()
start_time = time.perf_counter()

print("Importing")
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../dataset'))
from data_loader import DataLoader

from collaborative_filtering_model import CollaborativeFiltering  # noqa: F401
from common_neighbors_model import CommonNeighborsModel   # noqa: F401
from ncf_model import NCFModel    # noqa: F401
from popularity_model import GamePopularityModel    # noqa: F401

import tracemalloc
import time

model = GamePopularityModel()
file_name = None
N = 50


print("Initializing Data Loader")
data_loader = DataLoader()
network = data_loader.get_full_network()
data_loader.load_random_train_test_network(network=network, train_percentage=0.8, test_percentage=0.2, seed=0)
model.set_data_loader(data_loader)

try:
    model.load(file_name)
except NotImplementedError:
    model.train()
preds = model.recommend_n_games_for_user(76561198166465514, N)

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
