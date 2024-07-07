package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"strconv"
	"time"
)

func addInteractionHandler(response_writer http.ResponseWriter, request *http.Request, userID int64) {
	var url_values map[string]interface{}
	decoder := json.NewDecoder(request.Body)
	err := decoder.Decode(&url_values)
	if err != nil {
		log.Printf("Invalid request body.")
		writeErrorJSONResponse(response_writer, "Invalid request body.", http.StatusBadRequest)
		return
	}

	params := map[string]interface{}{
		// Model Params
		"rec_model_name":                 validateParameterStringPost("rec_model_name", response_writer, url_values, true),
		"rec_model_save_path":            validateParameterStringPost("rec_model_save_path", response_writer, url_values, true),
		"num_game_interactions_local":    validateParameterStringPostAndConvertToInt64("num_game_interactions_local", response_writer, url_values, true),
		"num_game_owned_local":           validateParameterStringPostAndConvertToInt64("num_game_owned_local", response_writer, url_values, true),
		"num_game_interactions_external": validateParameterStringPostAndConvertToInt64("num_game_interactions_external", response_writer, url_values, true),
		"num_game_owned_external":        validateParameterStringPostAndConvertToInt64("num_game_owned_external", response_writer, url_values, true),
		// Interaction Params
		"game_id":            validateParameterStringPostAndConvertToInt64("game_id", response_writer, url_values, true),
		"user_liked":         validateParameterStringPostAndConvertToBool("user_liked", response_writer, url_values, true),
		"time_spent":         validateParameterFloatPost("time_spent", response_writer, url_values, true),
		"steam_link_clicked": validateParameterStringPostAndConvertToBool("steam_link_clicked", response_writer, url_values, true),
	}
	for _, value := range params {
		if value == nil {
			return
		}
	}

	firestoreClient := getFirestoreClient()
	params["user_id"] = userID
	params["timestamp"] = time.Now().Unix()
	_, err = firestoreClient.Collection("interactions").Doc("data").Collection(strconv.FormatInt(params["user_id"].(int64), 10)).Doc(strconv.FormatInt(params["game_id"].(int64), 10)).Set(context.Background(), params)
	if err != nil {
		log.Printf("Failed to save interaction: " + err.Error())
		http.Error(response_writer, "Failed to save interaction: "+err.Error(), http.StatusInternalServerError)
		return
	}

	responseData := map[string]interface{}{
		"success": 1,
	}
	writeJSONResponse(response_writer, appendRequestMetaData(responseData, request))
}
