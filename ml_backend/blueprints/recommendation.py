from flask import Blueprint, current_app, g, jsonify
from auth import login_required
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
        "evaluation_test_common_neighbors_constant_scoring",
        "test_evaluation_constant_scoring",
        None,
    ),
    ModelWrapper(
        CommonNeighbors,
        "evaluation_test_common_neighbors_percentile_scoring",
        "test_evaluation_data_loader_percentile_scoring",
        None,
    ),
    ModelWrapper(
        CommonNeighbors,
        "evaluation_test_common_neighbors",
        "test_evaluation_data_loader",
        None,
    ),
    ModelWrapper(
        GamePopularityModel,
        "evaluation_test_popularity_model",
        "test_evaluation_data_loader",
        None,
    ),
    ModelWrapper(
        RandomModel,
        "evaluation_test_random_model",
        "test_evaluation_data_loader",
        None,
    ),
    ModelWrapper(
        NCFModel,
        "evaluation_test_cf_low_weight_decay_increased_lr_best_model_bugfix",
        "test_evaluation_data_loader",
        None,
    ),
    ModelWrapper(
        NCFModel,
        "evaluation_test_gcf_low_weight_decay_increased_lr_best_model_bugfix",
        "test_evaluation_data_loader",
        None,
    ),
    ModelWrapper(
        NCFModel,
        "evaluation_test_mlp_low_weight_decay_increased_lr_best_model_bugfix",
        "test_evaluation_data_loader",
        None,
    ),
    ModelWrapper(
        NCFModel,
        "evaluation_test_ncf_low_weight_decay_increased_lr_best_model_bugfix",
        "test_evaluation_data_loader",
        None,
    ),
    ModelWrapper(
        NCFModel,
        "evaluation_test_cf_embed_all_except_tags_genres_best_model_bugfix_clip_embeddings",
        "test_evaluation_data_loader_embed_all_except_tags_genres",
        None,
    ),
    ModelWrapper(
        NCFModel,
        "evaluation_test_gcf_embed_all_except_tags_genres_best_model_bugfix_clip_embeddings",
        "test_evaluation_data_loader_embed_all_except_tags_genres",
        None,
    ),
    ModelWrapper(
        NCFModel,
        "evaluation_test_mlp_embed_all_except_tags_genres_best_model_bugfix_clip_embeddings",
        "test_evaluation_data_loader_embed_all_except_tags_genres",
        None,
    ),
    ModelWrapper(
        NCFModel,
        "evaluation_test_ncf_embed_all_except_tags_genres_best_model_bugfix_clip_embeddings",
        "test_evaluation_data_loader_embed_all_except_tags_genres",
        None,
    ),
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
    print(g.execution_id, "Getting recommendations", model.name(), model_wrapper.model_save_file_name)
    if not data_loader.user_exists(g.user_id):
        return jsonify({"error": f"User with user_id {g.user_id} not found"}), 404
    model.fine_tune(g.user_id)
    recommendations = model.score_and_predict_n_games_for_user(
        g.user_id, query.N, should_sort=True
    )
    recommendations = [
        {**data_loader.get_game_information(game_id), "recommendation_score": float(score)}
        for game_id, score in recommendations
    ]

    # NOTE: This method assumes that the model trained on only all local data that is in the docker container & fine tuned on only all external data in the database.
    users_games_df = data_loader.get_users_games_df_for_user(g.user_id, preprocess=False)
    interactions_df = data_loader.get_interactions_df_for_user(g.user_id, preprocess=False)
    output = {
        "recommendations": recommendations,
        "model_name": model.name(),
        "model_save_path": model_wrapper.model_save_file_name,
        "num_game_interactions_local": len(interactions_df[interactions_df['source'] == LOCAL_DATA_SOURCE]),
        "num_game_owned_local": len(users_games_df[users_games_df['source'] == LOCAL_DATA_SOURCE]),
        "num_game_interactions_external": len(interactions_df[interactions_df['source'] == EXTERNAL_DATA_SOURCE]),
        "num_game_owned_external": len(users_games_df[users_games_df['source'] == EXTERNAL_DATA_SOURCE]),
    }
    print(g.execution_id, "Returning recommendations response")
    return jsonify(output)
