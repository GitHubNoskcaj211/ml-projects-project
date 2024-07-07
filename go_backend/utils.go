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
	"sync"
	"time"

	"cloud.google.com/go/firestore"
	firebase "firebase.google.com/go"
	"firebase.google.com/go/auth"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
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
			http.Error(response_writer, "Unauthorized", http.StatusUnauthorized)
			return
		}
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
		conf := &firebase.Config{ProjectID: "steam-game-recommender-415605"}
		_firebaseApp, err = firebase.NewApp(ctx, conf)
		if err != nil {
			log.Fatalln(err)
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

func acquireLock(lockPath string) error {
	client := getFirestoreClient()
	docRef := client.Collection("locks").Doc(lockPath)
	return client.RunTransaction(context.Background(), func(ctx context.Context, tx *firestore.Transaction) error {
		docSnap, err := tx.Get(docRef)
		if err != nil && status.Code(err) != codes.NotFound {
			return fmt.Errorf("transaction failed: %v", err)
		}
		if docSnap.Exists() {
			return fmt.Errorf("lock already acquired")
		}
		tx.Set(docRef, map[string]interface{}{
			"locked": true,
		})
		return nil
	})
}

func releaseLock(lockPath string) {
	client := getFirestoreClient()
	docRef := client.Collection("locks").Doc(lockPath)
	_, err := docRef.Delete(context.Background())
	if err != nil {
		log.Printf("Error releasing lock: %v", err)
	}
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

func mergeMaps(maps ...map[string]interface{}) map[string]interface{} {
	result := make(map[string]interface{})
	for _, m := range maps {
		for k, v := range m {
			result[k] = v
		}
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

func validateParameterString(key string, response_writer http.ResponseWriter, url_values url.Values, send_error bool) interface{} {
	value := url_values.Get(key)
	if value == "" {
		if send_error {
			writeErrorJSONResponse(response_writer, "Missing "+key+" parameter.", http.StatusBadRequest)
		}
		return nil
	}
	return value
}

func validateParameterStringList(key string, response_writer http.ResponseWriter, url_values url.Values, send_error bool) []string {
	values := url_values[key]
	if len(values) == 0 {
		if send_error {
			writeErrorJSONResponse(response_writer, "Missing "+key+" parameter.", http.StatusBadRequest)
		}
		return nil
	}
	return values
}

func validateParameterInt(key string, response_writer http.ResponseWriter, url_values url.Values, send_error bool) interface{} {
	value_str := validateParameterString(key, response_writer, url_values, send_error)
	if value_str == nil {
		return nil
	}
	log.Printf(value_str.(string))
	value_int, err := strconv.ParseInt(value_str.(string), 10, 64)
	if err != nil {
		if send_error {
			writeErrorJSONResponse(response_writer, "Parameter "+key+" was not an integer.", http.StatusBadRequest)
		}
		return nil
	}
	return value_int
}

func validateParameterIntList(key string, response_writer http.ResponseWriter, url_values url.Values, send_error bool) []int64 {
	values := validateParameterStringList(key, response_writer, url_values, send_error)
	if values == nil {
		return nil
	}

	intValues := make([]int64, len(values))
	for ii, value_str := range values {
		value_int, err := strconv.ParseInt(value_str, 10, 64)
		if err != nil {
			if send_error {
				writeErrorJSONResponse(response_writer, "Parameter "+key+" contained a non-integer value.", http.StatusBadRequest)
			}
			return nil
		}
		intValues[ii] = value_int
	}
	return intValues
}

func validateParameterStringPost(key string, response_writer http.ResponseWriter, url_values map[string]interface{}, send_error bool) interface{} {
	value := url_values[key]
	if value == nil {
		if send_error {
			writeErrorJSONResponse(response_writer, "Missing "+key+" parameter.", http.StatusBadRequest)
		}
		return nil
	}
	if _, ok := value.(string); !ok {
		if send_error {
			writeErrorJSONResponse(response_writer, key+" was not a string.", http.StatusBadRequest)
		}
		return nil
	}
	return value
}

func validateParameterFloatPost(key string, response_writer http.ResponseWriter, url_values map[string]interface{}, send_error bool) interface{} {
	value := url_values[key]
	if value == nil {
		if send_error {
			writeErrorJSONResponse(response_writer, "Missing "+key+" parameter.", http.StatusBadRequest)
		}
		return nil
	}
	if _, ok := value.(float64); !ok {
		if send_error {
			writeErrorJSONResponse(response_writer, key+" was not a float64.", http.StatusBadRequest)
		}
		return nil
	}
	return value
}

func validateParameterStringPostAndConvertToInt64(key string, response_writer http.ResponseWriter, url_values map[string]interface{}, send_error bool) interface{} {
	value := validateParameterStringPost(key, response_writer, url_values, send_error)
	if value == nil {
		return nil
	}
	value_int, err := strconv.ParseInt(value.(string), 10, 64)
	if err != nil {
		if send_error {
			writeErrorJSONResponse(response_writer, "Can't convert "+key+" to int64.", http.StatusBadRequest)
		}
		return nil
	}
	return value_int
}

func validateParameterStringPostAndConvertToBool(key string, response_writer http.ResponseWriter, url_values map[string]interface{}, send_error bool) interface{} {
	value := validateParameterStringPost(key, response_writer, url_values, send_error)
	if value == nil {
		return nil
	}
	value_bool, err := strconv.ParseBool(value.(string))
	if err != nil {
		if send_error {
			writeErrorJSONResponse(response_writer, "Can't convert "+key+" to bool.", http.StatusBadRequest)
		}
		return nil
	}
	return value_bool
}

const MAX_WAIT_TIME_SECONDS = 300

func makeAsyncRequest(url string) {
	go func() {
		client := &http.Client{
			Timeout: MAX_WAIT_TIME_SECONDS * time.Second,
		}
		_, err := client.Get(url)
		if err != nil {
			fmt.Println("Failed to make request:", err)
			return
		}
	}()
}
