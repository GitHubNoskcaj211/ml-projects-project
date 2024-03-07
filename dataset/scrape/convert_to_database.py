import pandas as pd
from constants import *
import os
import sqlite3


def convert_to_database(connection, file_name):
    csv_file = os.path.join(DATA_ROOT_DIR, file_name)
    df = pd.read_csv(csv_file)
    table_name = f'{file_name.split(".")[0]}'
    df.to_sql(table_name, connection, if_exists='replace', index=False)

def convert_all_to_database():
    global_database_file = os.path.join(DATA_ROOT_DIR, 'global_database.db')
    if os.path.exists(global_database_file):
        os.remove(global_database_file)
    connection = sqlite3.connect(os.path.join(DATA_ROOT_DIR, global_database_file))

    convert_to_database(connection, 'friends.csv')
    convert_to_database(connection, 'games.csv')
    convert_to_database(connection, 'invalids.csv')
    convert_to_database(connection, 'users_games.csv')
    convert_to_database(connection, 'users.csv')

    connection.commit()
    connection.close()