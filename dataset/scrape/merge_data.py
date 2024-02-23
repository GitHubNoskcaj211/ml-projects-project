import os
import pandas as pd
import sys

from constants import *


def merge(filename):
    snowballs = filter(lambda x: os.path.isdir(os.path.join(DATA_ROOT_DIR, x)), os.listdir(DATA_ROOT_DIR))
    snowball_files = map(lambda x: os.path.join(DATA_ROOT_DIR, x, filename), snowballs)

    output = pd.concat(map(pd.read_csv, snowball_files))
    match filename:
        case "users.csv" | "games.csv":
            output.drop_duplicates("id", inplace=True)
        case "friends.csv":
            output.drop_duplicates(inplace=True)
        case "users_games.csv":
            output.drop_duplicates(["user_id", "game_id"], inplace=True)
        case _:
            assert False
    output.to_csv(os.path.join(DATA_ROOT_DIR, filename), index=False)


if __name__ == "__main__":
    filename = os.path.basename(sys.argv[1])
    merge(filename)
