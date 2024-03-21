from flask import Blueprint, request, Response, redirect, current_app, jsonify
from flask_login import (
    current_user,
    LoginManager,
    login_user,
    login_required,
    logout_user,
)
import os
import pandas as pd
import shutil
import traceback
from urllib.parse import urlencode

from dataset.scrape.get_data import CACHE, ENVIRONMENT, FILE_MANAGER, get_single_user

steam_login = Blueprint(name="steam_login", import_name=__name__)
login_manager = LoginManager()


class User:
    def __init__(self, id):
        self.id = id
        self.is_authenticated = True
        self.is_active = True
        self.is_anonymous = False

    def get_id(self):
        return self.id


@login_manager.user_loader
def load_user(user_id):
    return User(user_id)


@steam_login.route("/login", methods=["GET"])
def auth_with_steam():
    # if len(request.args) == 0:
    #     params = {
    #         "openid.ns": "http://specs.openid.net/auth/2.0",
    #         "openid.identity": "http://specs.openid.net/auth/2.0/identifier_select",
    #         "openid.claimed_id": "http://specs.openid.net/auth/2.0/identifier_select",
    #         "openid.mode": "checkid_setup",
    #         "openid.return_to": f"{current_app.config['BACKEND_URL']}/login",
    #         "openid.realm": current_app.config["BACKEND_URL"],
    #     }
    #     auth_url = f"https://steamcommunity.com/openid/login?{urlencode(params)}"
    #     return redirect(auth_url)

    # user_url = request.args.get("openid.identity")
    # if user_url is None:
    #     return Response("Login failed", status=401)
    id = 76561198103368250#user_url[user_url.rfind("/") + 1:]
    assert login_user(User(id))
    resp = redirect(current_app.config["FRONTEND_URL"])
    return resp


@steam_login.route("/init_user", methods=["GET"])
@login_required
def init_user():
    if current_app.users_games_ref.document(current_user.id).get().exists:
        return jsonify(id=current_user.id)
    try:
        ENVIRONMENT.initialize_environment(
            current_app.config["STEAM_WEB_API_KEY"], current_user.id, None
        )
        shutil.rmtree(ENVIRONMENT.SNOWBALL_ROOT_DIR)
        os.mkdir(ENVIRONMENT.SNOWBALL_ROOT_DIR)
        FILE_MANAGER.open_files()
        success = get_single_user(current_user.id)
        FILE_MANAGER.close_files()
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        return Response("Bad Scrape", 500)
    if not success:
        print("Failed to scrape")
        return Response("Bad Scrape", 500)

    user_data_dir = os.path.join(ENVIRONMENT.DATA_ROOT_DIR, current_user.id)

    games = pd.read_csv(os.path.join(user_data_dir, "games.csv"))
    games = games.to_dict("records")

    friends = pd.read_csv(os.path.join(user_data_dir, "friends.csv"))
    assert (friends["user1"] == int(current_user.id)).all()
    friends = {"friends": friends.to_dict("records")}

    user_games = pd.read_csv(os.path.join(user_data_dir, "users_games.csv"))
    assert (user_games["user_id"] == int(current_user.id)).all()
    user_games = {"games": user_games.to_dict("records")}

    for game in games:
        current_app.games_ref.document(str(game["id"])).set(game, merge=True)
    current_app.friends_ref.document(current_user.id).set(friends, merge=True)
    current_app.users_games_ref.document(current_user.id).set(user_games, merge=True)

    return jsonify(id=current_user.id)


@steam_login.route("/logout", methods=["GET"])
@login_required
def logout():
    logout_user()
    return redirect(current_app.config["FRONTEND_URL"])
