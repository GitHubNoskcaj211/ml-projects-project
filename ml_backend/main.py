import os

if "K_SERVICE" in os.environ:
    import googlecloudprofiler
    googlecloudprofiler.start(service="backend")

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config
import time
import os
from flask import Flask, jsonify, g, request
from flask_cors import CORS
import uuid
from urllib.parse import urlparse, urlunparse
import ujson

from utils.firestore import DatabaseClient

from blueprints.errors import errors
from blueprints.recommendation import recommendation, model_wrappers

from dotenv import load_dotenv

load_dotenv()


def create_app():
    app = Flask(__name__)
    app.register_blueprint(errors)
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
        print(g.execution_id, os.getpid(), "ROUTE CALLED ", request.url)

    @app.after_request
    def after_request(response):
        if response and (data := response.get_json()) and isinstance(data, dict):
            data["time_request"] = int(time.time())
            data["execution_time_ms"] = int(time.time() * 1000 - g.start_time * 1000)
            data["version"] = config.VERSION
            data["name"] = config.NAME
            response.set_data(ujson.dumps(data))
        return response

    app.config.from_prefixed_env()
    app.debug = app.config["DEBUG"] == "development"

    return app


app = create_app()
app.default_data_loader = None
app.model_wrappers = model_wrappers

app.database_client = DatabaseClient()

if __name__ == "__main__":
    print("Starting app...")
    app.run(host="0.0.0.0", port=3001)
