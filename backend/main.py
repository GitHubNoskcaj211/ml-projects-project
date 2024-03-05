import sys

sys.path.append("..")

import os
import config
import json
import time
from flask import Flask, jsonify, g, request
from flask_cors import CORS
import uuid
from urllib.parse import urlparse, urlunparse

# from blueprints.activities import activities
from blueprints.errors import errors
from blueprints.games import games
from blueprints.steam_login import steam_login, login_manager
from blueprints.recommendation import recommendation, model_wrappers

from dotenv import load_dotenv

load_dotenv()


def create_app():
    app = Flask(__name__)
    app.register_blueprint(errors)
    app.register_blueprint(games)
    app.register_blueprint(steam_login)
    app.register_blueprint(recommendation)

    @app.errorhandler(404)
    def resource_not_found(e):
        return jsonify(error=str(e)), 404

    @app.errorhandler(405)
    def resource_not_found(e):
        return jsonify(error=str(e)), 405

    @app.route("/version", methods=["GET"], strict_slashes=False)
    def version():
        response_body = {
            "success": 1,
        }
        return jsonify(response_body)

    @app.before_request
    def before_request_func():
        execution_id = uuid.uuid4()
        g.start_time = time.time()
        g.execution_id = execution_id
        print(g.execution_id, "ROUTE CALLED ", request.url)

    @app.after_request
    def after_request(response):
        if response and response.get_json():
            data = response.get_json()
            data["time_request"] = int(time.time())
            data["execution_time_ms"] = int(time.time() * 1000 - g.start_time * 1000)
            data["version"] = config.VERSION
            response.set_data(json.dumps(data))
        return response

    app.config.from_prefixed_env()
    app.debug = app.config["DEBUG"] == "development"

    return app


app = create_app()
app.secret_key = bytes.fromhex(app.config["SECRET_KEY"])
app.data_loader = None
app.model_wrappers = model_wrappers
app.config["SESSION_COOKIE_SAMESITE"] = "None"
app.config["SESSION_COOKIE_SECURE"] = True

frontend_url_parsed = urlparse(app.config["FRONTEND_URL"])
frontend_url_parsed = frontend_url_parsed._replace(path="", params="", query="", fragment="")
origin = urlunparse(frontend_url_parsed)
cors = CORS(app, origins=[origin], supports_credentials=True)

login_manager.init_app(app)

if __name__ == "__main__":
    print("Starting app...")
    app.run(host="0.0.0.0", port=3000)
