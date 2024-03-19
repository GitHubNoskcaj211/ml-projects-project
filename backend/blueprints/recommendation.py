from flask import Blueprint, current_app, jsonify
from flask_login import current_user, login_required
from flask_pydantic import validate
from pydantic import BaseModel
from models.common_neighbors_model import CommonNeighborsModelStorageMemoryEfficient
from models.popularity_model import GamePopularityModel
from backend_utils.utils import (
    load_and_get_data_loader,
    load_and_get_random_model_wrapper,
    ModelWrapper,
)

recommendation = Blueprint(name="recommendation", import_name=__name__)

model_wrappers = [
    ModelWrapper(
        CommonNeighborsModelStorageMemoryEfficient,
        "test_common_neighbors_storage_memory_efficient",
        None,
    ),
    # ModelWrapper(GamePopularityModel, "test_popularity_model", None),
]


class GetRecommendationFilterInput(BaseModel, extra="forbid"):
    N: int


@recommendation.route("/get_N_recommendations_for_user", methods=["GET"])
@login_required
@validate()
def get_recommendations(query: GetRecommendationFilterInput):
    data_loader = load_and_get_data_loader(current_app)
    model_wrapper = load_and_get_random_model_wrapper(current_app)
    model = model_wrapper.model
    user_id = current_user.id
    if not data_loader.user_exists(user_id):
        return jsonify({"error": f"User with user_id {user_id} not found"}), 404
    model.fine_tune(user_id)
    recommendations = model.score_and_predict_n_games_for_user(
        user_id, query.N, should_sort=True
    )
    recommendations = [
        {"game_id": int(game_id), "recommendation_score": float(score)}
        for game_id, score in recommendations
    ]
    output = {
        "recommendations": recommendations,
        "model_name": model.name(),
        "model_save_path": model_wrapper.save_file_name,
    }
    return jsonify(output)


class Interaction(BaseModel, extra="forbid"):
    game_id: int
    user_liked: bool
    time_spent: float


@recommendation.route("/add_interaction", methods=["POST"])
@login_required
@validate()
def add_interaction(body: Interaction):
    interaction = body.model_dump()
    interaction["user_id"] = int(current_user.id)
    current_app.interactions_ref.document("data").collection(str(current_user.id)).document(str(interaction["game_id"])).set(interaction)
    return jsonify({"success": 1})
