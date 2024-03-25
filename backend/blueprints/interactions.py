from flask import Blueprint, current_app, jsonify
from flask_login import current_user, login_required
from flask_pydantic import validate
from pydantic import BaseModel
from backend_utils.utils import (
    load_and_get_data_loader,
)

interactions = Blueprint(name="interactions", import_name=__name__)


class Interaction(BaseModel, extra="forbid"):
    # Model Params
    rec_model_name: str
    rec_model_save_path: str
    # Interaction Params
    game_id: int
    user_liked: bool
    time_spent: float


@interactions.route("/add_interaction", methods=["POST"])
@login_required
@validate()
def add_interaction(body: Interaction):
    interaction = body.model_dump()
    interaction["user_id"] = int(current_user.id)
    current_app.database_client.interactions_ref \
        .document("data") \
        .collection(str(current_user.id)) \
        .document(str(interaction["game_id"])) \
        .set(interaction)
    return jsonify({"success": 1})


@interactions.route("/get_all_interactions_for_user", methods=["GET"])
@login_required
def get_recommendations():
    data_loader = load_and_get_data_loader(current_app)
    user_id = int(current_user.id)
    if not data_loader.user_exists(user_id):
        return jsonify({"error": f"User with user_id {user_id} not found"}), 404
    output = data_loader.get_interactions_df_for_user(user_id)
    output = {"interactions": output.to_dict("records")}
    return jsonify(output)
