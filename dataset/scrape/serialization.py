import io
import os
import pandas as pd
from tqdm import tqdm
import ujson

from dataset.scrape.constants import *


def serialize_users_games():
    csv_file = os.path.join(ENVIRONMENT.DATA_ROOT_DIR, "users_games.csv")
    df = pd.read_csv(csv_file, dtype={
        "user_id": "int64",
        "game_id": "int64",
        "playtime_forever": "int64",
    })
    groups = df.groupby("user_id")

    offsets = {}
    filepath = os.path.join(ENVIRONMENT.DATA_ROOT_DIR, "users_games.bin")
    with open(filepath, "wb") as f:
        for user_id, group in tqdm(groups):
            start = f.tell()
            group.to_feather(f)
            size = f.tell() - start
            offsets[user_id] = [start, size]

        offsets_str = ujson.dumps(offsets)
        f.seek(0)
        f.write(len(offsets_str).to_bytes(8, "little"))
        f.write(offsets_str.encode("utf-8"))


def deserialize_users_games(user_id):
    filepath = os.path.join(ENVIRONMENT.DATA_ROOT_DIR, "users_games.bin")
    user_id = str(user_id)
    with open(filepath, "rb") as f:
        offsets_len = int.from_bytes(f.read(8), "little")
        offsets = ujson.loads(f.read(offsets_len).decode("utf-8"))

        start, size = offsets.get(user_id, [None, None])
        if start is None:
            return None
        f.seek(start)
        data = io.BytesIO(f.read(size))
    return pd.read_feather(data)
