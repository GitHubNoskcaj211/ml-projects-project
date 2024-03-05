from collections import deque
import dataclasses
import pandas as pd

from constants import *
from merge_all import merge_all
from remove_zero_playtime_users import remove_zero_playtime_users

FILENAMES = [
    (USERS_FILENAME, User),
    (GAMES_FILENAME, Game),
    (FRIENDS_FILENAME, Friend),
    (USER_GAMES_FILENAME, UserGame),
    (INVALIDS_FILENAME, InvalidData),
]


def open_files():
    files = []
    for filename, dataclass in FILENAMES:
        f = open(filename, "a+", encoding="utf-8")
        if f.tell() == 0:
            header = ",".join(map(lambda x: x.name, dataclasses.fields(dataclass)))
            f.write(header + "\n")
        files.append(f)
    return tuple(files)


files = open_files()
users_f, games_f, friends_f, user_games_f, invalids_f = files
log_f = open(LOG_FILENAME, "a+", encoding="utf-8")
log_f.seek(0)


def close_files():
    print("Saving")
    assert len(FILENAMES)
    for file in files:
        file.close()
    log_f.close()

    print('Checking for Zero Playtime (Private) Users')
    remove_zero_playtime_users()

    print("Merging")
    merge_all()


def replay_log():
    visited_valid = set()
    user_ids = deque([])
    for line in log_f:
        line = line.strip()
        log_type, user_id = line.split(" ")
        log_type = LogType(int(log_type))
        match log_type:
            case LogType.ADD_QUEUE:
                user_ids.append(user_id)
            case LogType.VISITED_VALID:
                next_popped = user_ids.popleft()
                while next_popped != user_id:
                    next_popped = user_ids.popleft()
                visited_valid.add(user_id)
            case _:
                print("Invalid log type", log_type)
                assert False
    return user_ids, visited_valid


def get_invalids():
    if os.path.exists(ALL_INVALIDS_FILENAME):
        invalids = pd.read_csv(ALL_INVALIDS_FILENAME)
    else:
        invalids = None

    def extract():
        return (
            set(invalids[invalids["type"] == InvalidDataType.USER.value]["id"]),
            set(invalids[invalids["type"] == InvalidDataType.GAME.value]["id"]),
        )

    invalids_f.seek(0)
    invalids_f.readline()
    if invalids_f.read(1) == "":
        if invalids is None:
            return set(), set()
        return extract()

    if invalids is None:
        invalids = pd.read_csv(invalids_f)
    else:
        invalids = pd.concat([invalids, pd.read_csv(invalids_f)])

    return extract()


def get_parsed_games():
    if os.path.exists(ALL_GAMES_FILENAME):
        all_games = set(pd.read_csv(ALL_GAMES_FILENAME)["id"])
    else:
        all_games = set()

    games_f.seek(0)
    games_f.readline()
    if games_f.read(1) == "":
        return all_games
    games_f.seek(0)
    games_parsed = pd.read_csv(games_f)["id"]
    games_parsed_set = set(games_parsed)
    assert len(games_parsed_set) == len(games_parsed)

    games_parsed_set.update(all_games)

    return games_parsed_set


def write_log(log_type: LogType, user_id):
    log_f.write(f"{log_type.value} {user_id}\n")


def write_data(file, data):
    def convert(x):
        repl = str(x).replace('"', '""')
        return f'"{repl}"'
    line = ",".join(map(convert, dataclasses.astuple(data)))
    file.write(line + "\n")


def write_user(user: User):
    write_data(users_f, user)


def write_friend(friend: Friend):
    write_data(friends_f, friend)


def write_game(game: Game):
    write_data(games_f, game)


def write_user_game(user_game: UserGame):
    write_data(user_games_f, user_game)


def write_invalid(invalid_data: InvalidData):
    write_data(invalids_f, invalid_data)
