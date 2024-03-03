from flask import Blueprint, request, Response, redirect, current_app
from urllib.parse import urlencode

steam_login = Blueprint(name="steam_login", import_name=__name__)

@steam_login.route("/auth", methods=["GET"])
def auth_with_steam():
    redirect_uri = request.args.get("redirect_uri")
    if redirect_uri is None:
        return Response("redirect_uri query argument is required", status=400)
    params = {
        "openid.ns": "http://specs.openid.net/auth/2.0",
        "openid.identity": "http://specs.openid.net/auth/2.0/identifier_select",
        "openid.claimed_id": "http://specs.openid.net/auth/2.0/identifier_select",
        "openid.mode": "checkid_setup",
        "openid.return_to": redirect_uri,
        "openid.realm": current_app.config["URL"],
    }
    auth_url = f"https://steamcommunity.com/openid/login?{urlencode(params)}"
    return redirect(auth_url)