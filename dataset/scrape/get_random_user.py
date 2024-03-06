import random
import requests

from dataset.scrape.constants import *

START = 76561197960952452
END = 76561199616561598


def verify_request(URL):
    resp = requests.get(URL.format(user_id=ENVIRONMENT.ROOT))
    if resp.status_code == 401 or resp.status_code == 404:
        return None
    if resp.status_code != 200:
        print(resp.status_code)
        print(resp.text)
    assert resp.status_code == 200
    return resp.json()


while True:
    user_id = random.randint(START, END)

    resp = verify_request(ENVIRONMENT.FRIENDS_URL.format(user_id=user_id))
    if resp is None:
        continue
    if resp["friendslist"]["friends"] == []:
        continue

    resp = verify_request(ENVIRONMENT.GAMES_URL.format(user_id=user_id))
    if resp is None:
        continue
    resp = resp["response"]
    if "game_count" not in resp or resp["game_count"] == 0:
        continue
    if all(resp_game["playtime_forever"] == 0 for resp_game in resp["games"]):
        continue

    print(user_id)
    break
