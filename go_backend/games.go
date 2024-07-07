package main

import (
	"encoding/json"
	"net/http"

	"github.com/buger/jsonparser"
)

func getGameInformation(response_writer http.ResponseWriter, game_id string) map[string]interface{} {
	initGameData()

	gameBytes, _, _, err := jsonparser.Get(gameDataBytes, game_id)
	if err != nil {
		responseData := make(map[string]interface{})
		responseData["error"] = "game_id not found."
		response_writer.WriteHeader(http.StatusNotFound)
		writeJSONResponse(response_writer, responseData)
		return nil
	}

	var game_response map[string]interface{}
	if err := json.Unmarshal(gameBytes, &game_response); err != nil {
		writeErrorJSONResponse(response_writer, "Error unmarshaling game data", http.StatusInternalServerError)
		return nil
	}

	return game_response
}

func getGameInformationHandler(response_writer http.ResponseWriter, request *http.Request) {
	url_values := request.URL.Query()
	game_id := validateParameterString("game_id", response_writer, url_values, true)
	if game_id == nil {
		return
	}

	game_response := getGameInformation(response_writer, game_id.(string))
	if (game_response == nil) {
		return
	}
	writeJSONResponse(response_writer, appendRequestMetaData(game_response, request))
}
