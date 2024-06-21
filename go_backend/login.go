package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

const OPENID_NS = "http://specs.openid.net/auth/2.0"
const OPENID_URL = "https://steamcommunity.com/openid/login"

func getVerifyLoginURL() string {
	u := app.Config.BackendURL
	u.Path = "/verify_login"
	return u.String()
}

func loginHandler(response_writer http.ResponseWriter, request *http.Request) {
	params := url.Values{}
	params.Add("openid.ns", OPENID_NS)
	params.Add("openid.mode", "checkid_setup")
	params.Add("openid.claimed_id", "http://specs.openid.net/auth/2.0/identifier_select")
	params.Add("openid.identity", "http://specs.openid.net/auth/2.0/identifier_select")
	params.Add("openid.return_to", getVerifyLoginURL())
	params.Add("openid.realm", app.Config.BackendURL.String())

	url, err := url.Parse(OPENID_URL)
	if err != nil {
		writeErrorJSONResponse(response_writer, "Failed to construct auth url: "+err.Error(), http.StatusInternalServerError)
		return
	}
	url.RawQuery = params.Encode()
	http.Redirect(response_writer, request, url.String(), http.StatusFound)
}

func verifyLoginHandler(response_writer http.ResponseWriter, request *http.Request) {
	reqParams := request.URL.Query()

	if reqParams.Get("openid.ns") != OPENID_NS ||
		reqParams.Get("openid.mode") != "id_res" ||
		reqParams.Get("openid.op_endpoint") != OPENID_URL ||
		reqParams.Get("openid.return_to") != getVerifyLoginURL() {
		writeErrorJSONResponse(response_writer, "Invalid verify params", http.StatusBadRequest)
		return
	}

	params := reqParams
	params.Set("openid.mode", "check_authentication")
	resp, err := http.PostForm(OPENID_URL, params)
	if err != nil {
		writeErrorJSONResponse(response_writer, "Failed to verify login: "+err.Error(), http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		writeErrorJSONResponse(response_writer, "Failed to read verify response body: "+err.Error(), http.StatusInternalServerError)
		return
	}

	response := strings.Split(string(body), "\n")
	if len(response) != 3 || response[0] != "ns:"+OPENID_NS || response[1] != "is_valid:true" || response[2] != "" {
		writeErrorJSONResponse(response_writer, "Invalid verify response", http.StatusBadRequest)
		return
	}

	id_url := reqParams.Get("openid.claimed_id")
	if !strings.HasPrefix(id_url, "https://steamcommunity.com/openid/id/") {
		writeErrorJSONResponse(response_writer, "Invalid claimed_id", http.StatusBadRequest)
		return
	}
	userID := strings.TrimPrefix(id_url, "https://steamcommunity.com/openid/id/")
	if _, err := strconv.ParseInt(userID, 10, 64); err != nil {
		writeErrorJSONResponse(response_writer, "Invalid user id", http.StatusBadRequest)
		return
	}

	authClient := getAuthClient()
	token, err := authClient.CustomToken(context.Background(), userID)
	if err != nil {
		fmt.Println("Failed to create custom token: ", err)
		writeErrorJSONResponse(response_writer, "Failed to create custom token", http.StatusInternalServerError)
		return
	}
	url := fmt.Sprintf("%s?token=%s", app.Config.FrontendURL, token)
	http.Redirect(response_writer, request, url, http.StatusFound)
}

func makeAndUnmarshalRequest(response_writer http.ResponseWriter, request string) map[string]interface{} {
	resp, err := http.Get(request)
	if err != nil {
		writeErrorJSONResponse(response_writer, "Failed to fetch owned games.", http.StatusInternalServerError)
		return nil
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		writeErrorJSONResponse(response_writer, "Failed to read response body.", http.StatusInternalServerError)
		return nil
	}
	var result map[string]interface{}
	err = json.Unmarshal(body, &result)
	if err != nil {
		writeErrorJSONResponse(response_writer, "Failed to parse JSON response.", http.StatusInternalServerError)
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
						writeErrorJSONResponse(response_writer, "Friend steamid was not an integer", http.StatusBadRequest)
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

func initUserHandler(response_writer http.ResponseWriter, request *http.Request, userID int64) {
	// NOTE: games in firestore is deprecated (to ease requirements of init_user).
	// Instead, any new games should be scraped during the sync process.

	firestoreClient := getFirestoreClient()
	_, err := firestoreClient.Collection("users_games").Doc(strconv.FormatInt(userID, 10)).Get(context.Background())
	if err == nil {
		responseData := map[string]interface{}{
			"id": userID,
		}
		writeJSONResponse(response_writer, appendRequestMetaData(responseData, request))
		return
	}

	users_games_document_data := getUsersGamesDocumentData(response_writer, userID)
	if users_games_document_data == nil {
		return
	}

	friends_document_data := getFriendsDocumentData(response_writer, userID)
	if friends_document_data == nil {
		return
	}

	firestoreClient.Collection("users_games").Doc(strconv.FormatInt(userID, 10)).Set(context.Background(), users_games_document_data)
	firestoreClient.Collection("friends").Doc(strconv.FormatInt(userID, 10)).Set(context.Background(), friends_document_data)
}
