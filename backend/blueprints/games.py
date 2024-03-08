from flask import Blueprint, current_app, jsonify
from flask_pydantic import validate
from pydantic import BaseModel, Extra
from backend_utils.utils import load_and_get_data_loader

games = Blueprint(name="game", import_name=__name__)


class GetGameInformationFilterInput(BaseModel, extra=Extra.forbid):
    game_id: int


@games.route("/get_game_information", methods=["GET"])
@validate()
def get_game_information(query: GetGameInformationFilterInput):
    data_loader = load_and_get_data_loader(current_app)
    game_id = query.game_id
    info = data_loader.get_game_information(game_id)
    if info is None:
        return jsonify({"error": f"Game with game_id {game_id} not found"}), 404
    elif len(info) > 1:
        return jsonify({"error": f"Multiple games found for game_id {game_id}"}), 500
    return jsonify(info[0])
