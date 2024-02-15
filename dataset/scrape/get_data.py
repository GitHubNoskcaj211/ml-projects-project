import grequests
import requests
from tqdm import tqdm

from util import *

ROOT = "76561198166465514"  # Akash's steam id
NUM_USERS = 5000

games_parsed = get_parsed_games()


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
    resp = make_single_request(GAMES_URL, user_id=user_id)
    if resp is None:
        return False
    resp = resp.json()["response"]
    if len(resp) == 0 or resp["game_count"] == 0:
        return False
    resp_games = resp["games"]

    def add_edge(resp_game):
        write_user_game(UserGame(
            user_id=user_id,
            game_id=resp_game["appid"],
            playtime_2weeks=resp_game.get("playtime_2weeks", 0),
            playtime_forever=resp_game["playtime_forever"]
        ))

    valid = False
    requests = []
    new_resp_games = []
    for resp_game in resp_games:
        game_id = resp_game["appid"]
        if game_id in games_parsed:
            add_edge(resp_game)
            valid = True
        else:
            new_resp_games.append(resp_game)
            requests.append(grequests.get(GAME_DATA_URL.format(app_id=game_id)))
    num_already_seen = len(resp_games) - len(requests)
    with tqdm(total=len(resp_games), initial=num_already_seen, desc="Games", position=1, leave=False) as pbar:
        for i, resp in grequests.imap_enumerated(requests, size=10):
            pbar.update(1)
            resp_game = new_resp_games[i]
            game_id = resp_game["appid"]
            games_parsed.add(game_id)

            if resp.status_code == 500:
                continue
            assert resp.status_code == 200

            valid = True
            game_name = resp_game["name"].strip()
            resp = resp.json()
            write_game(Game(
                id=game_id,
                name=game_name,
                numReviews=resp["reviews"],
                avgReviewScore=resp["reviewScore"],
                price=resp["price"],
                genres=resp["genres"],
                tags=resp["tags"],
                description=resp["description"],
                numFollowers=resp.get("followers", 0)
            ))
            add_edge(resp_game)
    return valid


user_ids, visited_valid, visited_invalid = replay_log()


def add_queue(user_id):
    if user_id in user_ids or user_id in visited_valid or user_id in visited_invalid:
        return
    user_ids.append(user_id)
    write_log(LogType.ADD_QUEUE, user_id)


def add_visited(visited_type, visited_set, user_id):
    assert user_id not in user_ids and user_id not in visited_valid and user_id not in visited_invalid
    write_log(visited_type, user_id)
    visited_set.add(user_id)


def add_visited_valid(user_id):
    write_user(User(id=user_id))
    add_visited(LogType.VISITED_VALID, visited_valid, user_id)


def add_visited_invalid(user_id):
    add_visited(LogType.VISITED_INVALID, visited_invalid, user_id)


try:
    if len(user_ids) == 0:
        add_queue(ROOT)
    with tqdm(total=NUM_USERS, initial=len(visited_valid), desc="Users", position=0) as pbar:
        while len(user_ids) > 0 and len(visited_valid) < NUM_USERS:
            user_id = user_ids.pop()
            if user_id in visited_valid or user_id in visited_invalid:
                continue
            resp = make_single_request(FRIENDS_URL, user_id=user_id)
            if resp is None:
                add_visited_invalid(user_id)
                continue
            valid = write_user_data(user_id)
            if not valid:
                add_visited_invalid(user_id)
                continue
            add_visited_valid(user_id)
            friends = resp.json()["friendslist"]["friends"]
            for friend in friends:
                friend_id = friend["steamid"]
                add_queue(friend_id)
                write_friend(Friend(user1=user_id, user2=friend_id))
            pbar.update(1)
except AssertionError as e:
    print(e)
    print("Rate Limited")
    pass
except KeyboardInterrupt:
    print("Handling Keyboard Interrupt")
    pass
except Exception as e:
    print(e)
    print("Unknown")
    pass

close_files()
