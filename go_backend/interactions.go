package main

import (
	"context"
	"encoding/json"
	"net/http"
	"strconv"
	"time"
)

func addInteractionHandler(response_writer http.ResponseWriter, request *http.Request) {
	requireLogin()

	var url_values map[string]interface{}
	decoder := json.NewDecoder(request.Body)
	err := decoder.Decode(&url_values)
	if err != nil {
		writeErrorJSONResponse(response_writer, http.StatusBadRequest, "Invalid request body.")
		return
	}

	params := map[string]interface{}{
		// Model Params
		"rec_model_name":                 validateParameterString("rec_model_name", response_writer, url_values),
		"rec_model_save_path":            validateParameterString("rec_model_save_path", response_writer, url_values),
		"num_game_interactions_local":    validateParameterInt("num_game_interactions_local", response_writer, url_values),
		"num_game_owned_local":           validateParameterInt("num_game_owned_local", response_writer, url_values),
		"num_game_interactions_external": validateParameterInt("num_game_interactions_external", response_writer, url_values),
		"num_game_owned_external":        validateParameterInt("num_game_owned_external", response_writer, url_values),
		// Interaction Params
		"game_id":            validateParameterInt("game_id", response_writer, url_values),
		"user_liked":         validateParameterBool("user_liked", response_writer, url_values),
		"time_spent":         validateParameterFloat("time_spent", response_writer, url_values),
		"steam_link_clicked": validateParameterBool("steam_link_clicked", response_writer, url_values),
	}
	for _, value := range params {
		if value == nil {
			return
		}
	}

	initFirestoreClient()
	params["user_id"] = int64(101) // TODO fix after login
	params["timestamp"] = time.Now().Unix()
	_, err = firestoreClient.Collection("interactions").Doc("data").Collection(strconv.FormatInt(params["user_id"].(int64), 10)).Doc(strconv.FormatInt(params["game_id"].(int64), 10)).Set(context.Background(), params)
	if err != nil {
		http.Error(response_writer, "Failed to save interaction: "+err.Error(), http.StatusInternalServerError)
		return
	}

	responseData := map[string]interface{}{
		"success": 1,
	}
	writeJSONResponse(response_writer, appendRequestMetaData(responseData, request))
}
