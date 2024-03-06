from flask import Blueprint, request, Response, redirect, current_app, jsonify
from flask_login import (
    current_user,
    LoginManager,
    login_user,
    login_required,
    logout_user,
)
from urllib.parse import urlencode
import sys
import subprocess

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
    if len(request.args) == 0:
        params = {
            "openid.ns": "http://specs.openid.net/auth/2.0",
            "openid.identity": "http://specs.openid.net/auth/2.0/identifier_select",
            "openid.claimed_id": "http://specs.openid.net/auth/2.0/identifier_select",
            "openid.mode": "checkid_setup",
            "openid.return_to": f"{current_app.config['BACKEND_URL']}/login",
            "openid.realm": current_app.config["BACKEND_URL"],
        }
        auth_url = f"https://steamcommunity.com/openid/login?{urlencode(params)}"
        return redirect(auth_url)

    user_url = request.args.get("openid.identity")
    if user_url is None:
        return Response("Login failed", status=401)
    id = user_url[user_url.rfind("/") + 1:]
    assert login_user(User(id))
    resp = redirect(current_app.config["FRONTEND_URL"])
    return resp


@steam_login.route("/init_user", methods=["GET"])
@login_required
def init_user():
    if current_app.debug:
        cwd = "../dataset/scrape/"
    else:
        cwd = "./dataset/scrape/"
    ret = subprocess.run(
        [sys.executable, "get_data.py"],
        cwd=cwd,
        env={
            "STEAM_WEB_API_KEY": current_app.config["STEAM_WEB_API_KEY"],
            "ROOT_USER": current_user.id,
            "NUM_USERS": "1",
        }
    )
    print(ret)
    return Response("Hi")


@steam_login.route("/logout", methods=["GET"])
@login_required
def logout():
    logout_user()
    return redirect(current_app.config["FRONTEND_URL"])


@steam_login.route("/user", methods=["GET"])
@login_required
def get_user():
    return jsonify(id=current_user.id)
