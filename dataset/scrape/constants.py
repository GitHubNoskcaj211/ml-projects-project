from dotenv import load_dotenv
import os
from pathlib import Path
import dataclasses
from enum import Enum

load_dotenv()


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


class InvalidDataType(Enum):
    USER = 1
    GAME = 2


@dataclasses.dataclass
class InvalidData:
    type: int  # InvalidDataType
    id: str


class LogType(Enum):
    ADD_QUEUE = 1
    VISITED_VALID = 2


class Environment:
    def initialize_environment(self, key, root, num_users):
        self.KEY = key
        self.ROOT = root
        self.NUM_USERS = num_users
        self.DATA_ROOT_DIR = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "../data_files"
        )

        self.ALL_GAMES_FILENAME = os.path.join(self.DATA_ROOT_DIR, "games.csv")
        self.ALL_INVALIDS_FILENAME = os.path.join(self.DATA_ROOT_DIR, "invalids.csv")

        self.SNOWBALL_ROOT_DIR = os.path.join(self.DATA_ROOT_DIR, self.ROOT)
        Path(self.SNOWBALL_ROOT_DIR).mkdir(parents=True, exist_ok=True)

        self.USERS_FILENAME = os.path.join(self.SNOWBALL_ROOT_DIR, "users.csv")
        self.GAMES_FILENAME = os.path.join(self.SNOWBALL_ROOT_DIR, "games.csv")
        self.FRIENDS_FILENAME = os.path.join(self.SNOWBALL_ROOT_DIR, "friends.csv")
        self.USER_GAMES_FILENAME = os.path.join(
            self.SNOWBALL_ROOT_DIR, "users_games.csv"
        )
        self.INVALIDS_FILENAME = os.path.join(self.SNOWBALL_ROOT_DIR, "invalids.csv")
        self.LOG_FILENAME = os.path.join(self.SNOWBALL_ROOT_DIR, "log.txt")

        self.FRIENDS_URL = f"https://api.steampowered.com/ISteamUser/GetFriendList/v0001/?key={self.KEY}&steamid={{user_id}}&relationship=friend"
        self.GAMES_URL = f"https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/?key={self.KEY}&steamid={{user_id}}&include_appinfo=1&include_played_free_games=1&format=json"
        self.GAME_DATA_URL = "https://api.gamalytic.com/game/{app_id}"

        self.FILENAMES = [
            (self.USERS_FILENAME, User),
            (self.GAMES_FILENAME, Game),
            (self.FRIENDS_FILENAME, Friend),
            (self.USER_GAMES_FILENAME, UserGame),
            (self.INVALIDS_FILENAME, InvalidData),
        ]


ENVIRONMENT = Environment()
ENVIRONMENT.initialize_environment(
    os.getenv("STEAM_WEB_API_KEY"), os.getenv("ROOT_USER"), int(os.getenv("NUM_USERS"))
)
