from collections import deque
import pandas as pd
import requests
from tqdm import tqdm

from common import *

URL = f"http://api.steampowered.com/ISteamUser/GetFriendList/v0001/?key={KEY}&steamid={{user_id}}&relationship=friend"
ROOT = "76561198166465514"  # Akash's steam id
NUM_USERS = 5000


friend_pairs = []
visited = set()
private = set()
try:
    user_ids = deque([ROOT])
    with tqdm(total=NUM_USERS) as pbar:
        while len(user_ids) > 0 and len(visited) < NUM_USERS:
            user_id = user_ids.pop()
            if user_id in visited or user_id in private:
                continue
            resp = requests.get(URL.format(user_id=user_id))
            if resp.status_code == 401:
                private.add(user_id)
                continue
            visited.add(user_id)
            assert resp.status_code == 200
            friends = resp.json()["friendslist"]["friends"]
            for friend in friends:
                friend_id = friend["steamid"]
                user_ids.append(friend_id)
                friend_pairs.append({
                    "user1": user_id,
                    "user2": friend_id,
                })
            pbar.update(1)
except AssertionError:
    print("Rate Limited")
    pass


users = pd.DataFrame(data=visited, columns=["id"])
users.sort_values(by="id", inplace=True)
users.to_csv(USERS_FILENAME, index=False)

friend_pairs = pd.DataFrame(friend_pairs)
friend_pairs = friend_pairs[friend_pairs["user1"] < friend_pairs["user2"]]
friend_pairs.to_csv(FRIENDS_FILENAME, index=False)
