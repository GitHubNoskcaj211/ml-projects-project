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

from models.common_neighbors_model import CommonNeighbors   # noqa: F401
from models.ncf_model import NCFModel    # noqa: F401
from models.popularity_model import GamePopularityModel    # noqa: F401

model_dispatcher = {
    'common_neighbors': CommonNeighbors,
    'game_popularity': GamePopularityModel,
    'ncf': NCFModel,
}
 
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help = "Model name to use", choices=list(model_dispatcher.keys()), required=True)
parser.add_argument("-f", "--load_file_name", help = "Model save to load in", type=str, required=True)
parser.add_argument("-N", "--num_games_to_recommend", help = "Num games to recommend", type=int, default=50)
args = parser.parse_args()
test_user_id = 76561198835352289 # 76561198103368250

import tracemalloc
import time

model = model_dispatcher[args.model]()
file_name = args.load_file_name
N = args.num_games_to_recommend

print("Initializing Data Loader")
data_loader = DataLoader() # get_external_database=True
model.set_data_loader(data_loader)
print('Loading Model')
model.load(file_name)
print('Fine Tuning')
model.fine_tune(test_user_id)
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
