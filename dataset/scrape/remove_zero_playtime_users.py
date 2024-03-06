import os
import pandas as pd
import sys

from dataset.scrape.constants import *

# Users with private playtime data will have all their games be 0 playtime. We remove this to keep only well defined users in the dataset.
def remove_zero_playtime_users():
    snowballs = list(filter(lambda x: os.path.isdir(os.path.join(ENVIRONMENT.DATA_ROOT_DIR, x)), os.listdir(ENVIRONMENT.DATA_ROOT_DIR)))
    user_files = list(map(lambda x: os.path.join(ENVIRONMENT.DATA_ROOT_DIR, x, 'users.csv'), snowballs))
    user_game_files = list(map(lambda x: os.path.join(ENVIRONMENT.DATA_ROOT_DIR, x, 'users_games.csv'), snowballs))
    log_files = list(map(lambda x: os.path.join(ENVIRONMENT.DATA_ROOT_DIR, x, 'log.txt'), snowballs))

    for user_file, user_game_file, log_file in zip(user_files, user_game_files, log_files):
        user_file_df = pd.read_csv(user_file)
        user_game_file_df = pd.read_csv(user_game_file)
        log_df = pd.read_csv(log_file, header=None, delimiter="\s+", names=["value", "user_id"])

        sum_total_playtime = user_game_file_df.groupby('user_id')['playtime_forever'].sum()
        user_ids_with_zero_playtime = sum_total_playtime[sum_total_playtime == 0].index.tolist()
        assert len(user_ids_with_zero_playtime) == 0

if __name__ == '__main__':
    remove_zero_playtime_users()
