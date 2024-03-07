import random
import requests
from tqdm import tqdm

from dataset.scrape.file_manager import *

START = 76561197960952452
END = 76561199616561598
NUM_SAMPLES = 1000
URL = f"http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/?key={ENVIRONMENT.KEY}&steamid={{user_id}}&include_appinfo=1&format=json"

num_public = 0
for _ in tqdm(range(NUM_SAMPLES)):
    user_id = random.randint(START, END)
    resp = requests.get(URL.format(user_id=user_id))
    if resp.status_code == 401:
        continue
    assert resp.status_code == 200
    resp = resp.json()["response"]
    if len(resp) == 0 or resp["game_count"] == 0:
        continue
    num_public += 1

print(num_public / NUM_SAMPLES)
