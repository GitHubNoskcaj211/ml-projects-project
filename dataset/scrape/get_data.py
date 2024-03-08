import sys
import os

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import grequests
import json
import requests
from tqdm import tqdm
import traceback

from dataset.scrape.file_manager import *
from dataset.scrape.merge_all import merge_all
from dataset.scrape.convert_to_database import convert_to_database


class Cache:
    def __init__(self):
        self.games_parsed = get_parsed_games()
        self.invalid_users, self.invalid_games = get_invalids()
        if __name__ != "__main__":
            self.user_ids = deque([])
            self.visited_valid = set()

    def replay(self):
        self.user_ids, self.visited_valid = replay_log()


if __name__ == "__main__":
    ENVIRONMENT.initialize_environment(
        os.getenv("STEAM_WEB_API_KEY"),
        os.getenv("ROOT_USER"),
        int(os.getenv("NUM_USERS")),
    )
    FILE_MANAGER.open_files()
CACHE = Cache()


def make_single_request(url, **kwargs):
    resp = requests.get(url.format(**kwargs))
    if resp.status_code == 504:
        print("Gateway Timeout")
        return make_single_request(url, **kwargs)
    if resp.status_code == 401:
        return None
    if resp.status_code != 200:
        print()
        print(resp.status_code)
        print(resp.text)
        return make_single_request(url, **kwargs)
    assert resp.status_code == 200
    return resp


def write_user_data(user_id):
    resp = make_single_request(ENVIRONMENT.GAMES_URL, user_id=user_id)
    if resp is None:
        return False
    resp = resp.json()["response"]
    if len(resp) == 0 or resp["game_count"] == 0:
        return False
    resp_games = resp["games"]
    if all(resp_game["playtime_forever"] == 0 for resp_game in resp_games):
        return False

    def add_edge(resp_game):
        write_user_game(
            UserGame(
                user_id=user_id,
                game_id=resp_game["appid"],
                playtime_2weeks=resp_game.get("playtime_2weeks", 0),
                playtime_forever=resp_game["playtime_forever"],
            )
        )

    valid = False
    requests = []
    new_resp_games = []
    for resp_game in resp_games:
        game_id = resp_game["appid"]
        if game_id in CACHE.invalid_games:
            continue
        if game_id in CACHE.games_parsed:
            add_edge(resp_game)
            valid = True
        else:
            new_resp_games.append(resp_game)
            requests.append(
                grequests.get(ENVIRONMENT.GAME_DATA_URL.format(app_id=game_id))
            )
    num_already_seen = len(resp_games) - len(requests)
    with tqdm(
        total=len(resp_games),
        initial=num_already_seen,
        desc="Games",
        position=1,
        leave=False,
    ) as pbar:
        for i, resp in grequests.imap_enumerated(requests, size=10):
            pbar.update(1)
            resp_game = new_resp_games[i]
            game_id = resp_game["appid"]

            if resp.status_code == 500:
                add_invalid_game(game_id)
                continue
            assert resp.status_code == 200

            resp = json.loads(resp.content)
            try:
                game_data = Game(
                    id=game_id,
                    name=resp["name"],
                    numReviews=resp["reviews"],
                    avgReviewScore=resp["reviewScore"],
                    price=resp["price"],
                    genres=resp["genres"],
                    tags=resp["tags"],
                    description=resp["description"],
                    numFollowers=resp["followers"],
                )
            except KeyError:
                add_invalid_game(game_id)
                continue
            CACHE.games_parsed.add(game_id)
            valid = True
            write_game(game_data)
            add_edge(resp_game)
    return valid


def add_queue(user_id):
    if (
        user_id in CACHE.user_ids
        or user_id in CACHE.visited_valid
        or user_id in CACHE.invalid_users
    ):
        return
    CACHE.user_ids.append(user_id)
    write_log(LogType.ADD_QUEUE, user_id)


def add_visited(visited_type, visited_set, user_id):
    if __name__ == "__main__":
        assert (
            user_id not in CACHE.user_ids
            and user_id not in CACHE.visited_valid
            and user_id not in CACHE.invalid_users
        )
    write_log(visited_type, user_id)
    visited_set.add(user_id)


def add_visited_valid(user_id):
    write_user(User(id=user_id))
    add_visited(LogType.VISITED_VALID, CACHE.visited_valid, user_id)


def add_visited_invalid(user_id):
    write_invalid(InvalidData(type=InvalidDataType.USER.value, id=user_id))
    CACHE.invalid_users.add(user_id)


def add_invalid_game(game_id):
    write_invalid(InvalidData(type=InvalidDataType.GAME.value, id=game_id))
    CACHE.invalid_games.add(game_id)


def get_single_user(user_id):
    if user_id in CACHE.visited_valid or user_id in CACHE.invalid_users:
        return False
    resp = make_single_request(ENVIRONMENT.FRIENDS_URL, user_id=user_id)
    if resp is None:
        add_visited_invalid(user_id)
        return False
    valid = write_user_data(user_id)
    if not valid:
        add_visited_invalid(user_id)
        return False
    add_visited_valid(user_id)
    friends = resp.json()["friendslist"]["friends"]
    for friend in friends:
        friend_id = friend["steamid"]
        add_queue(friend_id)
        write_friend(Friend(user1=user_id, user2=friend_id))
    return True


def get_data():
    if len(CACHE.user_ids) == 0:
        add_queue(ENVIRONMENT.ROOT)
    with tqdm(
        total=ENVIRONMENT.NUM_USERS,
        initial=len(CACHE.visited_valid),
        desc="Users",
        position=0,
    ) as pbar:
        while (
            len(CACHE.user_ids) > 0 and len(CACHE.visited_valid) < ENVIRONMENT.NUM_USERS
        ):
            user_id = CACHE.user_ids.popleft()
            if not get_single_user(user_id):
                continue
            pbar.update(1)


if __name__ == "__main__":
    try:
        CACHE.replay()
        get_data()
    except AssertionError as e:
        print(e)
        print("Rate Limited")
    except KeyboardInterrupt:
        print("Handling Keyboard Interrupt")
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        print("Unknown")

    FILE_MANAGER.close_files()

    print("Merging")
    merge_all()

    print("Converting to database")
    convert_to_database()
