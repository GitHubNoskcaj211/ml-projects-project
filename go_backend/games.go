package main

import (
	"encoding/json"
	"net/http"

	"github.com/buger/jsonparser"
)

func getGameInformationHandler(response_writer http.ResponseWriter, request *http.Request) {
	url_values := request.URL.Query()
	game_id := validateParameterString("game_id", response_writer, url_values)
	if game_id == nil {
		return
	}

	initGameData()

	gameBytes, _, _, err := jsonparser.Get(gameDataBytes, game_id.(string))
	if err != nil {
		responseData := make(map[string]interface{})
		responseData["error"] = "game_id not found."
		response_writer.WriteHeader(http.StatusNotFound)
		writeJSONResponse(response_writer, responseData)
		return
	}

	var gamesResponse map[string]interface{}
	if err := json.Unmarshal(gameBytes, &gamesResponse); err != nil {
		writeErrorJSONResponse(response_writer, "Error unmarshaling game data", http.StatusInternalServerError)
		return
	}

	gamesResponse["success"] = true
	writeJSONResponse(response_writer, appendRequestMetaData(gamesResponse, request))
}
