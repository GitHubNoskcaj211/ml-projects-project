from flask import Blueprint, request, Response, current_app, g, jsonify
from flask_pydantic import validate
from pydantic import BaseModel, Extra
import pandas as pd

games = Blueprint(name="game", import_name=__name__)

class GetGameInformationFilterInput(BaseModel, extra=Extra.forbid):
    game_id: int

@games.route('/get_game_information', methods=['GET'])
@validate()
def get_game_information(query: GetGameInformationFilterInput):
    games_df = current_app.data_loader.games_df
    game_id = query.game_id
    query_result = games_df[games_df['id'] == game_id]
    if query_result.empty:
        return jsonify({'error': f'Game with game_id {game_id} not found'}), 404
    elif len(query_result) > 1:
        return jsonify({'error': f'Multiple games found for game_id {game_id}'}), 500
    game_dict = query_result.iloc[0].to_dict()
    return jsonify(game_dict)