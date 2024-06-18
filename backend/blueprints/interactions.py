from flask import Blueprint, current_app, jsonify
from flask_login import current_user, login_required
from flask_pydantic import validate
from pydantic import BaseModel
from backend_utils.utils import (
    load_and_get_data_loader,
)
import time

interactions = Blueprint(name="interactions", import_name=__name__)


class Interaction(BaseModel, extra="forbid"):
    # Model Params
    rec_model_name: str
    rec_model_save_path: str
    num_game_interactions_local: int
    num_game_owned_local: int
    num_game_interactions_external: int
    num_game_owned_external: int
    # Interaction Params
    game_id: int
    user_liked: bool
    time_spent: float
    steam_link_clicked: bool


@interactions.route("/add_interaction", methods=["POST"])
@login_required
@validate()
def add_interaction(body: Interaction):
    interaction = body.model_dump()
    interaction["user_id"] = int(current_user.id)
    interaction["timestamp"] = time.time()
    current_app.database_client.interactions_ref.document("data").collection(
        str(current_user.id)
    ).document(str(interaction["game_id"])).set(interaction)
    return jsonify({"success": 1})
