import csv
from dotenv import load_dotenv
import os

load_dotenv()

DATA_ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data_files")
KEY = os.getenv("STEAM_WEB_API_KEY")
FRIENDS_FILENAME = os.path.join(DATA_ROOT_DIR, "friends.csv")
USERS_FILENAME = os.path.join(DATA_ROOT_DIR, "users.csv")


def write_to_file(filename, arr):
    if len(arr) == 0:
        return
    with open(filename, "w", newline="") as f:
        print("Writing to", os.path.basename(filename))
        writer = csv.DictWriter(f, fieldnames=arr[0].keys())
        writer.writeheader()
        writer.writerows(arr)
