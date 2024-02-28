from dotenv import load_dotenv
import os
from pathlib import Path
import dataclasses
from enum import Enum

load_dotenv()

KEY = os.getenv("STEAM_WEB_API_KEY")
ROOT = os.getenv("ROOT_USER")
NUM_USERS = os.getenv("NUM_USERS")

DATA_ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data_files")

ALL_GAMES_FILENAME = os.path.join(DATA_ROOT_DIR, "games.csv")

SNOWBALL_ROOT_DIR = os.path.join(DATA_ROOT_DIR, ROOT)
Path(SNOWBALL_ROOT_DIR).mkdir(parents=True, exist_ok=True)

USERS_FILENAME = os.path.join(SNOWBALL_ROOT_DIR, "users.csv")
GAMES_FILENAME = os.path.join(SNOWBALL_ROOT_DIR, "games.csv")
FRIENDS_FILENAME = os.path.join(SNOWBALL_ROOT_DIR, "friends.csv")
USER_GAMES_FILENAME = os.path.join(SNOWBALL_ROOT_DIR, "users_games.csv")
LOG_FILENAME = os.path.join(SNOWBALL_ROOT_DIR, "log.txt")

FRIENDS_URL = f"http://api.steampowered.com/ISteamUser/GetFriendList/v0001/?key={KEY}&steamid={{user_id}}&relationship=friend"
GAMES_URL = f"http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/?key={KEY}&steamid={{user_id}}&include_appinfo=1&include_played_free_games=1&format=json"
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
    INVALID_GAME = 4
