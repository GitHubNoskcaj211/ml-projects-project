import os
import pandas as pd

REPO_NAME = "data_files_public"
ORIGINAL_ROOT = os.path.join(os.path.dirname(__file__), "..", "dataset", "data_files")
PUBLIC_ROOT = os.path.join(os.path.dirname(__file__), REPO_NAME)

if os.path.exists(REPO_NAME):
    os.system(f"git -C {REPO_NAME} pull")
else:
    os.system(f"git clone git@hf.co:datasets/steamgamerecommender/{REPO_NAME}")


users = list(pd.read_csv(os.path.join(ORIGINAL_ROOT, "users.csv"))["id"])
users_mapping = {user: i for i, user in enumerate(users)}
next_anon_id = len(users)


def get_anon_id(user_id):
    global next_anon_id

    assert isinstance(user_id, int)
    id = users_mapping.get(user_id, None)
    if id is not None:
        return id
    users_mapping[user_id] = next_anon_id
    next_anon_id += 1
    return next_anon_id - 1


def anonymize_and_write(snowball, name, user_cols):
    print("Anonymizing", snowball, name)
    df = pd.read_csv(os.path.join(ORIGINAL_ROOT, snowball, name))
    for col in user_cols:
        df[col] = df[col].map(get_anon_id)
    anon_snowball = str(get_anon_id(int(snowball)))
    df.to_csv(os.path.join(PUBLIC_ROOT, anon_snowball, name), index=False)


snowballs = next(os.walk(os.path.join(ORIGINAL_ROOT)))[1]
snowballs.append("")

for snowball in snowballs:
    if not snowball.isdigit() or os.listdir(os.path.join(ORIGINAL_ROOT, snowball)) == []:
        continue
    anon_snowball = str(get_anon_id(int(snowball)))
    os.makedirs(os.path.join(PUBLIC_ROOT, anon_snowball), exist_ok=True)
    anonymize_and_write(snowball, "friends.csv", ["user1", "user2"])
    anonymize_and_write(snowball, "users_games.csv", ["user_id"])
    anonymize_and_write(snowball, "games.csv", [])
