package main

import (
	"context"
	"net/http"
	"strconv"
)

func initUserHandler(response_writer http.ResponseWriter, request *http.Request) {
	requireLogin()
	initFirestoreClient()

	user_id := int64(101) // TODO fix after login
	_, err := firestoreClient.Collection("users_games").Doc(strconv.FormatInt(user_id, 10)).Get(context.Background())
	if err == nil {
		responseData := map[string]interface{}{
			"id": user_id,
		}
		writeJSONResponse(response_writer, appendRequestMetaData(responseData, request))
		return
	}

	// TODO execute ./
}
