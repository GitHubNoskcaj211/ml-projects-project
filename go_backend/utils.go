package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"sync"

	"cloud.google.com/go/firestore"
	firebase "firebase.google.com/go"
	"firebase.google.com/go/auth"
	"google.golang.org/api/option"
)

var (
	gameDataBytes     []byte
	gameDataBytesOnce sync.Once

	_firebaseApp         *firebase.App
	_firebaseAppOnce     sync.Once
	_authClient          *auth.Client
	_authClientOnce      sync.Once
	_firestoreClient     *firestore.Client
	_firestoreClientOnce sync.Once
)

func getEnv(key, fallback string) string {
	if value, exists := os.LookupEnv(key); exists {
		return value
	}
	return fallback
}

type authHandlerFunc func(http.ResponseWriter, *http.Request, int64)

func requireLogin(handler authHandlerFunc) http.HandlerFunc {
	return http.HandlerFunc(func(response_writer http.ResponseWriter, request *http.Request) {
		auth := request.Header.Get("Authorization")
		if !strings.HasPrefix(auth, "Bearer ") {
			http.Error(response_writer, "Unauthorized", http.StatusUnauthorized)
			return
		}
		token := strings.TrimPrefix(auth, "Bearer ")

		authClient := getAuthClient()
		verifiedToken, err := authClient.VerifyIDToken(context.Background(), token)
		if err != nil {
			fmt.Println("Error verifying token: ", err)
			http.Error(response_writer, "Unauthorized", http.StatusUnauthorized)
			return
		}
		fmt.Println("Verified token: ", verifiedToken)
		userID, err := strconv.ParseInt(verifiedToken.UID, 10, 64)
		if err != nil {
			http.Error(response_writer, "Unauthorized", http.StatusUnauthorized)
			return
		}
		handler(response_writer, request, userID)
	})
}

func getFirebaseApp() *firebase.App {
	_firebaseAppOnce.Do(func() {
		ctx := context.Background()
		var err error
		if app.Config.GoogleApplicationCredentials == "" {
			conf := &firebase.Config{ProjectID: "steam-game-recommender-415605"}
			_firebaseApp, err = firebase.NewApp(ctx, conf)
			if err != nil {
				log.Fatalln(err)
			}

		} else {
			sa := option.WithCredentialsFile(app.Config.GoogleApplicationCredentials)
			_firebaseApp, err = firebase.NewApp(ctx, nil, sa)
			if err != nil {
				log.Fatalln(err)
			}
		}
	})
	return _firebaseApp
}

func getAuthClient() *auth.Client {
	_authClientOnce.Do(func() {
		var err error
		_authClient, err = getFirebaseApp().Auth(context.Background())
		if err != nil {
			log.Fatalln(err)
		}
	})
	return _authClient
}

func getFirestoreClient() *firestore.Client {
	_firestoreClientOnce.Do(func() {
		var err error
		_firestoreClient, err = getFirebaseApp().Firestore(context.Background())
		if err != nil {
			log.Fatalln(err)
		}
	})
	return _firestoreClient
}

// TODO convert all log.Fatal to http.Error & make sure that it doesn't keep processing request
func initGameData() {
	gameDataBytesOnce.Do(func() {
		gamesJsonFile, err := os.Open(app.Config.RootFolder + "dataset/data_files/games.json")
		if err != nil {
			log.Fatal("Error opening file:", err)
		}
		defer gamesJsonFile.Close()

		gameDataBytes, err = io.ReadAll(gamesJsonFile)
		if err != nil {
			log.Fatal("Error reading JSON file:", err)
		}
	})
}

func beforeRequest(next http.Handler) http.Handler {
	return http.HandlerFunc(func(response_writer http.ResponseWriter, request *http.Request) {
		start := time.Now()
		log.Printf("%d ROUTE CALLED %s", os.Getpid(), request.URL)
		ctx := context.WithValue(request.Context(), "start_time", start)
		next.ServeHTTP(response_writer, request.WithContext(ctx))
	})
}

func appendRequestMetaData(responseData map[string]interface{}, request *http.Request) map[string]interface{} {
	start := request.Context().Value("start_time").(time.Time)
	executionTimeMs := time.Since(start).Milliseconds()
	extraData := make(map[string]interface{})
	extraData["time_request"] = time.Now().Unix()
	extraData["execution_time_ms"] = executionTimeMs
	extraData["version"] = app.Config.Version
	extraData["backend_name"] = app.Config.Name
	mergedData := mergeMaps(responseData, extraData)
	return mergedData
}

func mergeMaps(base map[string]interface{}, extra map[string]interface{}) map[string]interface{} {
	result := make(map[string]interface{})
	for k, v := range base {
		result[k] = v
	}
	for k, v := range extra {
		result[k] = v
	}

	return result
}

func writeJSONResponse(response_writer http.ResponseWriter, response interface{}) {
	response_writer.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(response_writer).Encode(response); err != nil {
		http.Error(response_writer, err.Error(), http.StatusInternalServerError)
	}
}

func writeErrorJSONResponse(response_writer http.ResponseWriter, message string, statusCode int) {
	responseData := make(map[string]interface{})
	responseData["error"] = message
	response_writer.WriteHeader(statusCode)
	writeJSONResponse(response_writer, responseData)
}

func getValueFromURLValues(url_values interface{}, key string) interface{} {
	switch v := url_values.(type) {
	case map[string]interface{}: // For POST request.
		if value, ok := v[key]; ok {
			return fmt.Sprintf("%v", value)
		}
	case url.Values: // For GET request.
		return v.Get(key)
	}
	return ""
}

func validateParameterString(key string, response_writer http.ResponseWriter, url_values interface{}) interface{} {
	value := getValueFromURLValues(url_values, key)
	if value == "" {
		writeErrorJSONResponse(response_writer, "Missing "+key+" parameter.", http.StatusBadRequest)
		return nil
	}
	return value
}

func validateParameterInt(key string, response_writer http.ResponseWriter, url_values interface{}) interface{} {
	value_str := validateParameterString(key, response_writer, url_values)
	if value_str == nil {
		return nil
	}
	value_int, err := strconv.ParseInt(value_str.(string), 10, 64)
	if err != nil {
		writeErrorJSONResponse(response_writer, "Parameter "+key+" was not an integer.", http.StatusBadRequest)
		return nil
	}
	return value_int
}

func validateParameterFloat(key string, response_writer http.ResponseWriter, url_values interface{}) interface{} {
	value_str := validateParameterString(key, response_writer, url_values)
	if value_str == nil {
		return nil
	}
	value_float, err := strconv.ParseFloat(value_str.(string), 64)
	if err != nil {
		writeErrorJSONResponse(response_writer, "Parameter "+key+" was not a float.", http.StatusBadRequest)
		return nil
	}
	return value_float
}

func validateParameterBool(key string, response_writer http.ResponseWriter, url_values interface{}) interface{} {
	value_str := validateParameterString(key, response_writer, url_values)
	if value_str == nil {
		return nil
	}
	value_bool, err := strconv.ParseBool(value_str.(string))
	if err != nil {
		writeErrorJSONResponse(response_writer, "Parameter "+key+" was not a boolean.", http.StatusBadRequest)
		return nil
	}
	return value_bool
}
