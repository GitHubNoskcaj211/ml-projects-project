from collections import deque
import requests

from common import *

URL = f"http://api.steampowered.com/ISteamUser/GetFriendList/v0001/?key={KEY}&steamid={{user_id}}&relationship=friend"
ROOT = "76561198166465514"  # Akash's steam id
FRIENDS_FILENAME = os.path.join(DATA_ROOT_DIR, "friends.csv")
USERS_FILENAME = os.path.join(DATA_ROOT_DIR, "users.csv")


friend_pairs = []
visited = set()
try:
    user_ids = deque([ROOT])
    while len(user_ids) > 0 and len(visited) < 5:
        user_id = user_ids.pop()
        if user_id in visited:
            continue
        visited.add(user_id)
        resp = requests.get(URL.format(user_id=user_id))
        assert resp.status_code == 200
        friends = resp.json()["friendslist"]["friends"]
        for friend in friends:
            friend_id = friend["steamid"]
            user_ids.append(friend_id)
            friend_pairs.append((user_id, friend_id))
except AssertionError:
    # Rate limited
    pass


users = sorted(list(visited))
with open(USERS_FILENAME, "w") as f:
    for user in users:
        f.write(user + "\n")

friend_pairs = list(filter(lambda x: x[0] < x[1], friend_pairs))
with open(FRIENDS_FILENAME, "w") as f:
    for friend_pair in friend_pairs:
        f.write(",".join(friend_pair) + "\n")
