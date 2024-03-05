import grequests
import json
import requests
from tqdm import tqdm
import traceback

from util import *

games_parsed = get_parsed_games()
invalid_users, invalid_games = get_invalids()


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
    if all(resp_game["playtime_forever"] == 0 for resp_game in resp_games):
        return False

    def add_edge(resp_game):
        write_user_game(UserGame(
            user_id=user_id,
            game_id=resp_game["appid"],
            playtime_2weeks=resp_game.get("playtime_2weeks", 0),
            playtime_forever=resp_game["playtime_forever"]
        ))

    valid = False
    game_data_requests = []
    store_data_requests = []
    new_resp_games = []
    for resp_game in resp_games:
        game_id = resp_game["appid"]
        if game_id in invalid_games:
            continue
        if game_id in games_parsed:
            add_edge(resp_game)
            valid = True
        else:
            new_resp_games.append(resp_game)
            game_data_requests.append(grequests.get(GAME_DATA_URL.format(app_id=game_id)))
            store_data_requests.append(grequests.get(STORE_DATA_URL.format(app_id=game_id)))
    num_already_seen = len(resp_games) - len(new_resp_games)
    games_data = []
    games_valid = []
    for ii in range(len(new_resp_games)):
        games_data.append(Game(None, None, None, None, None, None, None, None, None, None))
        games_valid.append(True)
    with tqdm(total=len(resp_games), initial=num_already_seen, desc="Gamalytic", position=1, leave=False) as pbar:
        for i, resp in grequests.imap_enumerated(game_data_requests, size=10):
            pbar.update(1)
            
            resp_game = new_resp_games[i]
            game_id = resp_game["appid"]
            game_data = games_data[i]
            game_data.id = game_id

            if resp.status_code == 500:
                games_valid[i] = False
                continue
            assert resp.status_code == 200

            resp = json.loads(resp.content)
            try:
                game_data.name = resp["name"]
                game_data.numReviews = resp["reviews"]
                game_data.avgReviewScore = resp["reviewScore"]
                game_data.price = resp["price"]
                game_data.genres = resp["genres"]
                game_data.tags = resp["tags"]
                game_data.description = resp["description"]
                game_data.numFollowers = resp["followers"]
            except KeyError:
                games_valid[i] = False
                continue
    with tqdm(total=len(resp_games), initial=num_already_seen, desc="Steam Store", position=1, leave=False) as pbar:
        for i, resp in grequests.imap_enumerated(store_data_requests, size=10):
            pbar.update(1)

            game_data = games_data[i]
            resp_game = new_resp_games[i]
            game_id = resp_game["appid"]
            assert resp is not None, f'Request is None. Game {game_id}.'
            assert resp.status_code == 200, f'Request not successful. Game {game_id}. Status {resp.status_code}'
            resp = json.loads(resp.content)
            try:
                if not resp[f'{game_id}']['success']:
                    assert not games_valid[i], f'Not Successful. Gamalytic says good, steam says bad {game_id}' # TODO just for testing
                    games_valid[i] = False
                    continue
                if resp[f'{game_id}']['data']['type'] != 'game':
                    assert not games_valid[i], f'Not game. Gamalytic says good, steam says bad {game_id}' # TODO just for testing
                    games_valid[i] = False
                    continue
                game_data.requiredAge = resp[f'{game_id}']['data']['required_age']
            except KeyError:
            assert not games_valid[i], f'Key Error. Gamalytic says good, steam says bad {game_id}' # TODO just for testing
                games_valid[i] = False
                continue
            import time
            time.sleep(0.125)
    for resp_game, game_data, game_valid in zip(new_resp_games, games_data, games_valid):
        if game_valid:
            game_id = resp_game["appid"]
            games_parsed.add(game_id)
            valid = True
            write_game(game_data)
            add_edge(resp_game)
        else:
            add_invalid_game(game_id)
    return valid


user_ids, visited_valid = replay_log()


def add_queue(user_id):
    if user_id in user_ids or user_id in visited_valid or user_id in invalid_users:
        return
    user_ids.append(user_id)
    write_log(LogType.ADD_QUEUE, user_id)


def add_visited(visited_type, visited_set, user_id):
    assert user_id not in user_ids and user_id not in visited_valid and user_id not in invalid_users
    write_log(visited_type, user_id)
    visited_set.add(user_id)


def add_visited_valid(user_id):
    write_user(User(id=user_id))
    add_visited(LogType.VISITED_VALID, visited_valid, user_id)


def add_visited_invalid(user_id):
    write_invalid(InvalidData(type=InvalidDataType.USER.value, id=user_id))
    invalid_users.add(user_id)


def add_invalid_game(game_id):
    write_invalid(InvalidData(type=InvalidDataType.GAME.value, id=game_id))
    invalid_games.add(game_id)


try:
    if len(user_ids) == 0:
        add_queue(ROOT)
    with tqdm(total=NUM_USERS, initial=len(visited_valid), desc="Users", position=0) as pbar:
        while len(user_ids) > 0 and len(visited_valid) < NUM_USERS:
            user_id = user_ids.popleft()
            if user_id in visited_valid or user_id in invalid_users:
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
except KeyboardInterrupt:
    print("Handling Keyboard Interrupt")
except Exception as e:
    print(e)
    print(traceback.format_exc())
    print("Unknown")

close_files()
