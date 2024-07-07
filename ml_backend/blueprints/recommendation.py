from flask import Blueprint, current_app, g, jsonify
from flask_pydantic import validate
from pydantic import BaseModel
from models.common_neighbors_model import CommonNeighbors
from models.popularity_model import GamePopularityModel
from models.ncf_model import NCFModel
from dataset.data_loader import EXTERNAL_DATA_SOURCE, LOCAL_DATA_SOURCE
from backend_utils.auth import login_required
from backend_utils.utils import (
    load_and_get_data_loader,
    load_and_get_model_wrapper,
    ModelWrapper,
)
from firebase_admin import firestore

recommendation = Blueprint(name="recommendation", import_name=__name__)

model_wrappers = {
    "evaluation_test_common_neighbors_constant_scoring":
        ModelWrapper(
            CommonNeighbors,
            "evaluation_test_common_neighbors_constant_scoring",
            "test_evaluation_constant_scoring",
            None,
        ),
    "evaluation_test_common_neighbors_percentile_scoring": 
        ModelWrapper(
            CommonNeighbors,
            "evaluation_test_common_neighbors_percentile_scoring",
            "test_evaluation_data_loader_percentile_scoring",
            None,
        ),
    "evaluation_test_common_neighbors": 
        ModelWrapper(
            CommonNeighbors,
            "evaluation_test_common_neighbors",
            "test_evaluation_data_loader",
            None,
        ),
    "evaluation_test_popularity_model": 
        ModelWrapper(
            GamePopularityModel,
            "evaluation_test_popularity_model",
            "test_evaluation_data_loader",
            None,
        ),
    "evaluation_test_cf_low_weight_decay_increased_lr_best_model_bugfix": 
        ModelWrapper(
            NCFModel,
            "evaluation_test_cf_low_weight_decay_increased_lr_best_model_bugfix",
            "test_evaluation_data_loader",
            None,
        ),
    "evaluation_test_gcf_low_weight_decay_increased_lr_best_model_bugfix": 
        ModelWrapper(
            NCFModel,
            "evaluation_test_gcf_low_weight_decay_increased_lr_best_model_bugfix",
            "test_evaluation_data_loader",
            None,
        ),
    "evaluation_test_mlp_low_weight_decay_increased_lr_best_model_bugfix": 
        ModelWrapper(
            NCFModel,
            "evaluation_test_mlp_low_weight_decay_increased_lr_best_model_bugfix",
            "test_evaluation_data_loader",
            None,
        ),
    "evaluation_test_ncf_low_weight_decay_increased_lr_best_model_bugfix": 
        ModelWrapper(
            NCFModel,
            "evaluation_test_ncf_low_weight_decay_increased_lr_best_model_bugfix",
            "test_evaluation_data_loader",
            None,
        ),
    "evaluation_test_cf_embed_all_except_tags_genres_best_model_bugfix_clip_embeddings": 
        ModelWrapper(
            NCFModel,
            "evaluation_test_cf_embed_all_except_tags_genres_best_model_bugfix_clip_embeddings",
            "test_evaluation_data_loader_embed_all_except_tags_genres",
            None,
        ),
    "evaluation_test_gcf_embed_all_except_tags_genres_best_model_bugfix_clip_embeddings": 
        ModelWrapper(
            NCFModel,
            "evaluation_test_gcf_embed_all_except_tags_genres_best_model_bugfix_clip_embeddings",
            "test_evaluation_data_loader_embed_all_except_tags_genres",
            None,
        ),
    "evaluation_test_mlp_embed_all_except_tags_genres_best_model_bugfix_clip_embeddings": 
        ModelWrapper(
            NCFModel,
            "evaluation_test_mlp_embed_all_except_tags_genres_best_model_bugfix_clip_embeddings",
            "test_evaluation_data_loader_embed_all_except_tags_genres",
            None,
        ),
    "evaluation_test_ncf_embed_all_except_tags_genres_best_model_bugfix_clip_embeddings": 
        ModelWrapper(
            NCFModel,
            "evaluation_test_ncf_embed_all_except_tags_genres_best_model_bugfix_clip_embeddings",
            "test_evaluation_data_loader_embed_all_except_tags_genres",
            None,
        ),
}


class RefreshRecommendationsQueueFilterInput(BaseModel, extra="forbid"):
    N: int
    model_save_file_name: str

def acquire_lock(lock_key):
    lock_ref = current_app.database_client.locks_ref.document(lock_key)
    @firestore.transactional
    def transaction_callback(transaction, lock_doc_ref):
        lock_doc = lock_doc_ref.get(transaction=transaction)
        if not lock_doc.exists:
            transaction.set(lock_doc_ref, {'locked': True})
            return True
        return False
    try:
        transaction = current_app.database_client.db.transaction()
        result = transaction_callback(transaction, lock_ref)
        if result:
            return True
        else:
            return False
    except Exception as e:
        print(f"Failed to acquire lock: {e}")
        return False

def release_lock(lock_key):
    lock_ref = current_app.database_client.locks_ref.document(lock_key)
    lock_ref.delete()

@recommendation.route("/refresh_recommendation_queue", methods=["GET"])
@login_required
@validate()
def refresh_recommendations_queue(query: RefreshRecommendationsQueueFilterInput):
    if query.model_save_file_name not in model_wrappers.keys():
        return jsonify({"error": f"Model with safe file name {query.model_save_file_name} not found"}), 404
    lock_key = f"refreshing:{g.user_id}:{query.model_save_file_name}"
    if not acquire_lock(lock_key):
        return jsonify({"error": "Another process is currently refreshing the recommendation queue for this user and model. Please try again later."}), 409

    try:
        data_loader = load_and_get_data_loader(current_app)
        model_wrapper = load_and_get_model_wrapper(model_wrappers[query.model_save_file_name])
        model = model_wrapper.model
        print(g.execution_id, "Getting recommendations", model.name(), model_wrapper.model_save_file_name)
        if not data_loader.user_exists(g.user_id):
            return jsonify({"error": f"User with user_id {g.user_id} not found"}), 404
        model.fine_tune(g.user_id)
        recommendations = model.score_and_predict_n_games_for_user(
            g.user_id, query.N, should_sort=True
        )
        
        # NOTE: This method assumes that the model trained on only all local data that is in the docker container & fine tuned on only all external data in the database.
        users_games_df = data_loader.get_users_games_df_for_user(g.user_id, preprocess=False)
        interactions_df = data_loader.get_interactions_df_for_user(g.user_id, preprocess=False)

        metadata_document_data = {
            'model_name': model.name(),
            'model_save_path': model_wrapper.model_save_file_name,
            'num_game_interactions_local': len(interactions_df[interactions_df['source'] == LOCAL_DATA_SOURCE]),
            'num_game_owned_local': len(users_games_df[users_games_df['source'] == LOCAL_DATA_SOURCE]),
            'num_game_interactions_external': len(interactions_df[interactions_df['source'] == EXTERNAL_DATA_SOURCE]),
            'num_game_owned_external': len(users_games_df[users_games_df['source'] == EXTERNAL_DATA_SOURCE]),
        }
        recommendation_queue_document_data = {
            'recommendations': [{'game_id': game_id, 'recommendation_score': score}for game_id, score in recommendations]
        }

        current_app.database_client.recommendation_queue_ref.document(str(g.user_id)).collection(query.model_save_file_name).document('recommendation_metadata').set(metadata_document_data)
        current_app.database_client.recommendation_queue_ref.document(str(g.user_id)).collection(query.model_save_file_name).document('recommendation_queue').set(recommendation_queue_document_data)

        print(g.execution_id, "Returning success.")
        return jsonify({"success": 1})
    finally:
        release_lock(lock_key)
