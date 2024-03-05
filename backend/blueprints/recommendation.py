from flask import Blueprint, request, Response, current_app, g, jsonify
from flask_pydantic import validate
from pydantic import BaseModel, Extra
import pandas as pd
from models.common_neighbors_model import CommonNeighborsModelStorageMemoryEfficient
from models.popularity_model import GamePopularityModel
from backend_utils.utils import load_and_get_random_model_wrapper, load_and_get_data_loader, ModelWrapper

recommendation = Blueprint(name="recommendation", import_name=__name__)

model_wrappers = [
    ModelWrapper(CommonNeighborsModelStorageMemoryEfficient, 'test_common_neighbors_storage_memory_efficient', None),
    ModelWrapper(GamePopularityModel, 'test_popularity_model', None)
]

class GetRecommendationFilterInput(BaseModel, extra=Extra.forbid):
    user_id: int
    N: int

@recommendation.route('/get_N_recommendations_for_user', methods=['GET'])
@validate()
def get_game_information(query: GetRecommendationFilterInput):
    data_loader = load_and_get_data_loader(current_app)
    model_wrapper = load_and_get_random_model_wrapper(current_app)
    model = model_wrapper.model

    if data_loader.users_df[data_loader.users_df['id'] == query.user_id].empty:
        return jsonify({'error': f'User with user_id {query.user_id} not found'}), 404
    recommendations = model.score_and_predict_n_games_for_user(query.user_id, query.N, should_sort=True)
    recommendations = [{'game_id': int(game_id), 'recommendation_score': float(score)} for game_id, score in recommendations]
    output = {'recommendations': recommendations, 'model_name': model.name(), 'model_save_path': model_wrapper.save_file_name}
    return jsonify(output)