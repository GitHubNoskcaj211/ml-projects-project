from collections import deque
import dataclasses
from dotenv import load_dotenv
from enum import Enum
import os
import pandas as pd

load_dotenv()

DATA_ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data_files")
KEY = os.getenv("STEAM_WEB_API_KEY")
USERS_FILENAME = os.path.join(DATA_ROOT_DIR, "users.csv")
GAMES_FILENAME = os.path.join(DATA_ROOT_DIR, "games.csv")
FRIENDS_FILENAME = os.path.join(DATA_ROOT_DIR, "friends.csv")
USER_GAMES_FILENAME = os.path.join(DATA_ROOT_DIR, "users_games.csv")
LOG_FILENAME = os.path.join(DATA_ROOT_DIR, "log.txt")

FRIENDS_URL = f"http://api.steampowered.com/ISteamUser/GetFriendList/v0001/?key={KEY}&steamid={{user_id}}&relationship=friend"
GAMES_URL = f"http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/?key={KEY}&steamid={{user_id}}&include_appinfo=1&format=json"
GAME_DATA_URL = "https://api.gamalytic.com/game/{app_id}"


@dataclasses.dataclass
class User:
    id: str


@dataclasses.dataclass
class Friend:
    user1: str
    user2: str


@dataclasses.dataclass
class Game:
    id: str
    name: str
    numReviews: int
    avgReviewScore: float
    price: float
    genres: list
    tags: list
    description: str
    numFollowers: int


@dataclasses.dataclass
class UserGame:
    user_id: str
    game_id: str
    playtime_2weeks: int
    playtime_forever: int


class LogType(Enum):
    ADD_QUEUE = 1
    VISITED_VALID = 2
    VISITED_INVALID = 3


FILENAMES = [
    (USERS_FILENAME, User),
    (GAMES_FILENAME, Game),
    (FRIENDS_FILENAME, Friend),
    (USER_GAMES_FILENAME, UserGame),
]


NUM_USERS = 5000


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
users_f, games_f, friends_f, user_games_f = files
log_f = open(LOG_FILENAME, "a+", encoding="utf-8")
log_f.seek(0)


def close_files():
    print("Saving")
    assert len(FILENAMES)
    for file in files:
        file.close()
    log_f.close()


def replay_log():
    visited_valid = set()
    visited_invalid = set()
    user_ids = deque([])
    for line in log_f:
        line = line.strip()
        log_type, user_id = line.split(" ")
        log_type = LogType(int(log_type))
        match log_type:
            case LogType.ADD_QUEUE:
                user_ids.append(user_id)
            case LogType.VISITED_VALID:
                a = user_ids.pop()
                assert a == user_id
                visited_valid.add(user_id)
            case LogType.VISITED_INVALID:
                assert user_ids.pop() == user_id
                visited_invalid.add(user_id)
            case _:
                print("Invalid log type", log_type)
                assert False
    return user_ids, visited_valid, visited_invalid


def get_parsed_games():
    games_f.seek(0)
    games_f.readline()
    if games_f.read(1) == "":
        return set()
    games_f.seek(0)
    games_parsed = pd.read_csv(games_f)["id"]
    games_parsed_set = set(games_parsed)
    assert len(games_parsed_set) == len(games_parsed)
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