package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
)

func makeAndUnmarshalRequest(response_writer http.ResponseWriter, request string) map[string]interface{} {
	resp, err := http.Get(request)
	if err != nil {
		writeErrorJSONResponse(response_writer, http.StatusInternalServerError, "Failed to fetch owned games.")
		return nil
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		writeErrorJSONResponse(response_writer, http.StatusInternalServerError, "Failed to read response body.")
		return nil
	}
	var result map[string]interface{}
	err = json.Unmarshal(body, &result)
	if err != nil {
		writeErrorJSONResponse(response_writer, http.StatusInternalServerError, "Failed to parse JSON response.")
		return nil
	}
	return result
}

func getUsersGamesDocumentData(response_writer http.ResponseWriter, user_id int64) map[string]interface{} {
	ownedGamesRequest := fmt.Sprintf("http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/?key=%s&steamid=%s&include_appinfo=1&include_played_free_games=1&format=json", app.Config.SteamWebAPIKey, strconv.FormatInt(user_id, 10))
	result := makeAndUnmarshalRequest(response_writer, ownedGamesRequest)
	if result == nil {
		return nil
	}
	users_games_document_data := make(map[string]interface{})
	users_games_document_data["synced"] = false
	if response, ok := result["response"].(map[string]interface{}); ok {
		if games, ok := response["games"].([]interface{}); ok {
			gamesSlice := make([]map[string]interface{}, len(games))
			for ii, game := range games {
				if gameMap, ok := game.(map[string]interface{}); ok {
					gamesSlice[ii] = make(map[string]interface{})
					gamesSlice[ii]["game_id"] = gameMap["appid"]
					if value, ok := gameMap["playtime_2weeks"]; ok {
						gamesSlice[ii]["playtime_2weeks"] = value
					} else {
						gamesSlice[ii]["playtime_2weeks"] = 0
					}
					gamesSlice[ii]["playtime_forever"] = gameMap["playtime_forever"]
					gamesSlice[ii]["user_id"] = user_id
				}
			}
			users_games_document_data["games"] = gamesSlice
		}
	}
	return users_games_document_data
}

func getFriendsDocumentData(response_writer http.ResponseWriter, user_id int64) map[string]interface{} {
	friendsRequest := fmt.Sprintf("http://api.steampowered.com/ISteamUser/GetFriendList/v0001/?key=%s&steamid=%s&relationship=friend", app.Config.SteamWebAPIKey, strconv.FormatInt(user_id, 10))
	result := makeAndUnmarshalRequest(response_writer, friendsRequest)
	if result == nil {
		return nil
	}
	friends_document_data := make(map[string]interface{})
	friends_document_data["synced"] = false
	if response, ok := result["friendslist"].(map[string]interface{}); ok {
		if friends, ok := response["friends"].([]interface{}); ok {
			friendsSlice := make([]map[string]interface{}, len(friends))
			for ii, friend := range friends {
				if friendMap, ok := friend.(map[string]interface{}); ok {
					friendsSlice[ii] = make(map[string]interface{})
					friendsSlice[ii]["user1"] = user_id
					value_int, err := strconv.ParseInt(friendMap["steamid"].(string), 10, 64)
					if err != nil {
						writeErrorJSONResponse(response_writer, http.StatusBadRequest, "Friend steamid was not an integer")
						return nil
					}
					friendsSlice[ii]["user2"] = value_int
				}
			}
			friends_document_data["friends"] = friendsSlice
		}
	}
	return friends_document_data
}

func initUserHandler(response_writer http.ResponseWriter, request *http.Request) {
	requireLogin()
	initFirestoreClient()

	// NOTE: games in firestore is deprecated (to ease requirements of init_user).
	// Instead, any new games should be scraped during the sync process.

	user_id := int64(101) // TODO fix after login
	_, err := firestoreClient.Collection("users_games").Doc(strconv.FormatInt(user_id, 10)).Get(context.Background())
	if err == nil {
		responseData := map[string]interface{}{
			"id": user_id,
		}
		writeJSONResponse(response_writer, appendRequestMetaData(responseData, request))
		return
	}

	users_games_document_data := getUsersGamesDocumentData(response_writer, user_id)
	if users_games_document_data == nil {
		return
	}

	friends_document_data := getFriendsDocumentData(response_writer, user_id)
	if friends_document_data == nil {
		return
	}

	firestoreClient.Collection("users_games").Doc(strconv.FormatInt(user_id, 10)).Set(context.Background(), users_games_document_data)
	firestoreClient.Collection("friends").Doc(strconv.FormatInt(user_id, 10)).Set(context.Background(), friends_document_data)
	responseData := map[string]interface{}{
		"id": user_id,
	}
	writeJSONResponse(response_writer, appendRequestMetaData(responseData, request))
}
