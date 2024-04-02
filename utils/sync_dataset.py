if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(os.path.join(os.path.dirname(__file__), "../dataset/scrape"))

from google.cloud.firestore_v1.base_query import FieldFilter
import os

from dataset.scrape.get_data import FILE_MANAGER, ENVIRONMENT, write_user, write_game, write_user_game, write_friend, User, Friend, Game, UserGame
from dataset.scrape.merge_all import merge_all
from dataset.scrape.convert_to_database import convert_all_to_database
import firestore


def sync(ref, write_ref_fn):
    query = ref.where(filter=FieldFilter("synced", "==", False))
    for data in query.stream():
        write_ref_fn(data)
        ref.document(data.id).update({"synced": True})


def main():
    users = set()

    def write_game_ref(data):
        data = data.to_dict()
        del data["synced"]
        write_game(Game(**data))

    def write_friend_ref(data):
        data = data.to_dict()
        for friend in data["friends"]:
            write_friend(Friend(**friend))

    def write_user_game_ref(data):
        users.add(data.id)
        data = data.to_dict()
        for user_game in data["games"]:
            write_user_game(UserGame(**user_game))

    client = firestore.DatabaseClient()
    print("Syncing Games")
    sync(client.games_ref, write_game_ref)
    print("Syncing Friends")
    sync(client.friends_ref, write_friend_ref)
    print("Syncing Users Games")
    sync(client.users_games_ref, write_user_game_ref)
    print("Syncing Users")
    for user in users:
        write_user(User(id=user))

    print("Merging All")
    merge_all()

    print("Converting all to Database")
    convert_all_to_database()


if __name__ == "__main__":
    ENVIRONMENT.initialize_environment(os.getenv("FLASK_STEAM_WEB_API_KEY"), "firestore", None)
    FILE_MANAGER.open_files()
    try:
        main()
    except Exception as e:
        print(e)
        pass
    FILE_MANAGER.close_files()
