from collections import deque
import requests
from tqdm import tqdm

from common import *

URL = f"http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/?key={KEY}&steamid={{user_id}}&include_appinfo=1&format=json"


with open(USERS_FILENAME, newline="") as f:
    f.readline()
    users = f.read().splitlines()

games = set()
user_games = []
try:
    for user_id in tqdm(users):
        resp = requests.get(URL.format(user_id=user_id))
        if resp.status_code == 401:
            continue
        resp = resp.json()["response"]
        if len(resp) == 0 or resp["game_count"] == 0:
            continue
        resp_games = resp["games"]
        for resp_game in resp_games:
            game_id = str(resp_game["appid"])
            game_name = resp_game["name"].strip()
            games.add((game_id, game_name))
            user_games.append({
                "user_id": user_id,
                "game_id": game_id,
                "playtime_2weeks": resp_game.get("playtime_2weeks"),
                "playtime_forever": resp_game["playtime_forever"],
            })
except AssertionError:
    print("Rate Limited")
    pass


games = sorted(list(games), key=lambda x: x[0])
games = list(map(lambda x: {"id": x[0], "name": x[1]}, games))
write_to_file(GAMES_FILENAME, games)

write_to_file(USER_GAMES_FILENAME, user_games)
