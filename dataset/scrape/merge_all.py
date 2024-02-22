from merge_data import merge


def merge_all():
    merge("users.csv")
    merge("games.csv")
    merge("friends.csv")
    merge("users_games.csv")


if __name__ == "__main__":
    merge_all()
