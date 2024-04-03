from flask import Blueprint, current_app, jsonify
from flask_login import current_user, login_required
from flask_pydantic import validate
from pydantic import BaseModel
from models.common_neighbors_model import CommonNeighbors
from models.popularity_model import GamePopularityModel
from models.random_model import RandomModel
from models.ncf_model import NCFModel
from dataset.data_loader import EXTERNAL_DATA_SOURCE, LOCAL_DATA_SOURCE
from backend_utils.utils import (
    load_and_get_data_loader,
    load_and_get_random_model_wrapper,
    ModelWrapper,
)

recommendation = Blueprint(name="recommendation", import_name=__name__)

model_wrappers = [
    ModelWrapper(
        CommonNeighbors,
        "test_common_neighbors_default",
        "test_common_neighbors_default_data_loader",
        None,
    ),
    ModelWrapper(
        CommonNeighbors,
        "test_common_neighbors_playtime_scored_gaussian_normalized",
        "test_common_neighbors_playtime_scored_gaussian_normalized_data_loader",
        None,
    ),
    ModelWrapper(
        GamePopularityModel,
        "test_popularity_model",
        "test_popularity_model_data_loader",
        None,
    ),
    ModelWrapper(
        RandomModel,
        "test_random_model",
        "test_random_model_data_loader",
        None,
    ),
    ModelWrapper(
        NCFModel,
        "test_cf_model",
        "test_ncf_data_loader",
        None,
    ),
    ModelWrapper(
        NCFModel,
        "test_gcf_model",
        "test_ncf_data_loader",
        None,
    ),
    ModelWrapper(
        NCFModel,
        "test_mlp_model",
        "test_ncf_data_loader",
        None,
    ),
    ModelWrapper(
        NCFModel,
        "test_ncf_model",
        "test_ncf_data_loader",
        None,
    ),
]


class GetRecommendationFilterInput(BaseModel, extra="forbid"):
    N: int


@recommendation.route("/get_N_recommendations_for_user", methods=["GET"])
@login_required
@validate()
def get_recommendations(query: GetRecommendationFilterInput):
    print("Getting recommendations")
    data_loader = load_and_get_data_loader(current_app)
    model_wrapper = load_and_get_random_model_wrapper(current_app)
    model = model_wrapper.model
    user_id = int(current_user.id)
    if not data_loader.user_exists(user_id):
        return jsonify({"error": f"User with user_id {user_id} not found"}), 404
    model.fine_tune(user_id)
    recommendations = model.score_and_predict_n_games_for_user(
        user_id, query.N, should_sort=True
    )
    recommendations = [
        {**data_loader.get_game_information(game_id), "recommendation_score": float(score)}
        for game_id, score in recommendations
    ]

    # NOTE: This method assumes that the model trained on only all local data that is in the docker container & fine tuned on only all external data in the database.
    users_games_df = data_loader.get_users_games_df_for_user(user_id, preprocess=False)
    interactions_df = data_loader.get_interactions_df_for_user(user_id, preprocess=False)
    output = {
        "recommendations": recommendations,
        "model_name": model.name(),
        "model_save_path": model_wrapper.model_save_file_name,
        "num_game_interactions_local": len(interactions_df[interactions_df['source'] == LOCAL_DATA_SOURCE]),
        "num_game_owned_local": len(users_games_df[users_games_df['source'] == LOCAL_DATA_SOURCE]),
        "num_game_interactions_external": len(interactions_df[interactions_df['source'] == EXTERNAL_DATA_SOURCE]),
        "num_game_owned_external": len(users_games_df[users_games_df['source'] == EXTERNAL_DATA_SOURCE]),
    }
    print("Returning recommendations response")
    return jsonify(output)
