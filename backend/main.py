from flask import Flask, redirect, request, Response
from urllib.parse import urlencode
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config.from_prefixed_env()
app.debug = app.config["ENV"] == "development"


@app.route("/auth", methods=["GET"])
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
        "openid.realm": app.config["URL"],
    }
    auth_url = f"https://steamcommunity.com/openid/login?{urlencode(params)}"
    return redirect(auth_url)


app.run(host="0.0.0.0", port=5000)
