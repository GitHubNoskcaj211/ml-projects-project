from functools import wraps
from flask import g, jsonify, request
from firebase_admin import auth


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get("Authorization")
        if token is None or not token.startswith("Bearer "):
            return jsonify({"error": "Unauthorized"}), 401
        token = token.split(" ")[1]
        try:
            decoded_token = auth.verify_id_token(token)
            g.user_id = int(decoded_token["uid"])
        except (ValueError, auth.InvalidIdTokenError, auth.ExpiredIdTokenError, auth.RevokedIdTokenError, auth.UserDisabledError):
            return jsonify({"error": "Unauthorized"}), 401
        except auth.CertificateFetchError:
            return jsonify({"error": "Unable to fetch the public key"}), 500

        return f(*args, **kwargs)
    return decorated_function
