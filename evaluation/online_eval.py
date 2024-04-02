if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import itertools
import pandas as pd
from utils.firestore import DatabaseClient


client = DatabaseClient()


def get_filtered_interactions(include_fn):
    users = client.interactions_ref.document("data").collections()

    def get_interactions_for_user(user_collection):
        return (interaction.to_dict() for interaction in user_collection.stream())
    interactions = (get_interactions_for_user(user_collection) for user_collection in users)
    return filter(lambda x: include_fn(x), itertools.chain.from_iterable(interactions))


def eval(include_fn, score_fn):
    data = pd.DataFrame(get_filtered_interactions(include_fn))
    data["expected_score"] = data.apply(score_fn, axis=1)
    data.rename(columns={
        "user_liked": "expected_edge",
        "user_id": "user",
        "game_id": "game",
    }, inplace=True)
    data["user_predicted_rank"] = data.groupby("user")["timestamp"].rank()
    data.sort_values("timestamp", inplace=True)

    return data


def include_coldstart(interaction):
    return interaction["num_game_interactions_local"] == 0 or interaction["num_game_owned_local"] == 0


def score_time_spent(interaction):
    return interaction["time_spent"]


print(eval(include_coldstart, score_time_spent))
