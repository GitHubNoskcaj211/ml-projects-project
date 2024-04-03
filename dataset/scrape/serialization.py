import ast
import io
import os
import pandas as pd
from tqdm import tqdm
import ujson

if __name__ == "__main__":
    import sys
    sys.path.append("../..")

from dataset.scrape.constants import *


def serialize_users_games():
    csv_file = os.path.join(ENVIRONMENT.DATA_ROOT_DIR, "users_games.csv")
    df = pd.read_csv(
        csv_file,
        dtype={
            "user_id": "int64",
            "game_id": "int64",
            "playtime_forever": "int64",
        },
    )
    groups = df.groupby("user_id")

    offsets = {}
    filepath = os.path.join(ENVIRONMENT.DATA_ROOT_DIR, "users_games.bin")
    data = io.BytesIO()
    for user_id, group in tqdm(groups):
        start = data.tell()
        group.to_feather(data)
        size = data.tell() - start
        offsets[user_id] = [start, size]
    with open(filepath, "wb") as f:
        offsets_str = ujson.dumps(offsets).encode("utf-8")
        f.write(len(offsets_str).to_bytes(8, "little"))
        f.write(offsets_str)
        f.write(data.getvalue())


USERS_GAMES_OFFSETS = None
USERS_GAMES_DATA = None


def deserialize_users_games(user_id):
    global USERS_GAMES_OFFSETS
    global USERS_GAMES_DATA

    if USERS_GAMES_DATA is None:
        filepath = os.path.join(ENVIRONMENT.DATA_ROOT_DIR, "users_games.bin")
        with open(filepath, "rb") as f:
            offsets_len = int.from_bytes(f.read(8), "little")
            USERS_GAMES_OFFSETS = ujson.loads(f.read(offsets_len).decode("utf-8"))
            USERS_GAMES_DATA = bytearray(f.read())
    user_id = str(user_id)
    ret = USERS_GAMES_OFFSETS.get(user_id, None)
    if ret is None:
        return None
    start, size = ret
    return pd.read_feather(io.BytesIO(USERS_GAMES_DATA[start:start + size]))


def serialize_games():
    csv_file = os.path.join(ENVIRONMENT.DATA_ROOT_DIR, "games.csv")
    df = pd.read_csv(
        csv_file,
        index_col="id",
        dtype={
            "id": "int64",
            "name": "string",
            "numReviews": "int64",
            "avgReviewScore": "int64",
            "price": "float64",
            "numFollowers": "int64",
        },
    )
    df["id"] = df.index
    df["genres"] = df["genres"].apply(ast.literal_eval)
    df["tags"] = df["tags"].apply(ast.literal_eval)
    data = df.to_dict("index")

    filepath = os.path.join(ENVIRONMENT.DATA_ROOT_DIR, "games.json")
    with open(filepath, "w") as f:
        ujson.dump(data, f)


GAMES = None


def deserialize_game(game_id):
    global GAMES

    if GAMES is None:
        filepath = os.path.join(ENVIRONMENT.DATA_ROOT_DIR, "games.json")
        with open(filepath, "r") as f:
            GAMES = ujson.load(f)
    return GAMES.get(str(game_id), None)


def serialize_all():
    serialize_users_games()
    serialize_games()
