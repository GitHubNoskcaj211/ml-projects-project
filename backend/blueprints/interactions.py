from flask import Blueprint, current_app, jsonify
from flask_login import current_user, login_required
from backend_utils.utils import (
    load_and_get_data_loader,
)

interactions = Blueprint(name="interactions", import_name=__name__)

@interactions.route("/get_all_interactions_for_user", methods=["GET"])
@login_required
def get_recommendations():
    data_loader = load_and_get_data_loader(current_app)
    user_id = current_user.id
    if not data_loader.user_exists(user_id):
        return jsonify({"error": f"User with user_id {user_id} not found"}), 404
    output = data_loader.get_interactions_df_for_user(user_id)
    output = {'interactions': output.to_dict('records')}
    return jsonify(output)