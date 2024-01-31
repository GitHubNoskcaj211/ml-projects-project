import pandas as pd
import requests
from tqdm import tqdm

from common import *

URL = f"http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/?key={KEY}&steamid={{user_id}}&include_appinfo=1&format=json"
GAME_URL = "https://api.gamalytic.com/game/{app_id}"


users = pd.read_csv(USERS_FILENAME)

games = set()
user_games = []
try:
    for user_id in tqdm(users["id"]):
        resp = requests.get(URL.format(user_id=user_id))
        if resp.status_code == 401:
            continue
        assert resp.status_code == 200
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

user_games = pd.DataFrame(user_games)
user_games.to_csv(USER_GAMES_FILENAME, index=False)


games_data = []
skipped = 0
for app_id, name in tqdm(games):
    resp = requests.get(GAME_URL.format(app_id=app_id))
    if resp.status_code == 500:
        skipped += 1
        continue
    assert resp.status_code == 200
    resp = resp.json()
    games_data.append({
        "id": app_id,
        "name": name,
        "numReviews": resp["reviews"],
        "avgReviewScore": resp["reviewScore"],
        "price": resp["price"],
        "genres": resp["genres"],
        "tags": resp["tags"],
        "numFollowers": resp.get("followers"),
    })

print("Skipped", skipped)
games_data = pd.DataFrame(data=games_data)
games_data.sort_values(by="id", inplace=True)
games_data.to_csv(GAMES_FILENAME, index=False)
