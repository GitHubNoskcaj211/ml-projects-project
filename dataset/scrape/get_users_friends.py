from collections import deque
import requests

from common import *

URL = f"http://api.steampowered.com/ISteamUser/GetFriendList/v0001/?key={KEY}&steamid={{user_id}}&relationship=friend"
ROOT = "76561198166465514"  # Akash's steam id


friend_pairs = []
visited = set()
try:
    user_ids = deque([ROOT])
    while len(user_ids) > 0 and len(visited) < 100:
        user_id = user_ids.pop()
        if user_id in visited:
            continue
        visited.add(user_id)
        resp = requests.get(URL.format(user_id=user_id))
        if resp.status_code == 401:
            continue
        assert resp.status_code == 200
        friends = resp.json()["friendslist"]["friends"]
        for friend in friends:
            friend_id = friend["steamid"]
            user_ids.append(friend_id)
            friend_pairs.append({
                "user1": user_id,
                "user2": friend_id,
            })
except AssertionError:
    print("Rate Limited")
    pass


users = list(map(lambda x: {"id": x}, sorted(visited)))
write_to_file(USERS_FILENAME, users)

friend_pairs = list(filter(lambda x: x["user1"] < x["user2"], friend_pairs))
write_to_file(FRIENDS_FILENAME, friend_pairs)
