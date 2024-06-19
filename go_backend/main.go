package main

// Note: To run use `go run *.go` in go_backend folder.

import (
	"log"
	"net/http"
	"strings"

	"github.com/gorilla/handlers"
	"github.com/gorilla/mux"
	"github.com/joho/godotenv"
)

type Config struct {
	Debug                        string
	SecretKey                    string
	FrontendURL                  string
	GoogleApplicationCredentials string
	Version                      string
	Name                         string
	RootFolder                   string
}

type App struct {
	Router *mux.Router
	Config Config
}

var app App

func main() {
	godotenv.Load()

	app = App{}
	app.Config = Config{
		Debug:                        getEnv("DEBUG", "production"),
		SecretKey:                    getEnv("SECRET_KEY", ""),
		FrontendURL:                  getEnv("FRONTEND_URL", ""),
		GoogleApplicationCredentials: getEnv("GOOGLE_APPLICATION_CREDENTIALS", ""),
		Version:                      getEnv("VERSION", ""),
		Name:                         getEnv("NAME", ""),
		RootFolder:                   getEnv("ROOT_FOLDER", ""),
	}

	app.Router = mux.NewRouter()
	app.Router.Use(beforeRequest)

	registerRoutes(app.Router)

	frontendURL := strings.TrimRight(app.Config.FrontendURL, "/")
	cors := handlers.CORS(
		handlers.AllowedOrigins([]string{frontendURL}),
		handlers.AllowCredentials(),
	)

	log.Println("Starting server on :3000...")
	log.Fatal(http.ListenAndServe(":3000", cors(app.Router)))
}

func versionHandler(response_writer http.ResponseWriter, request *http.Request) {
	responseData := map[string]interface{}{
		"success": true,
	}
	writeJSONResponse(response_writer, appendRequestMetaData(responseData, request))
}

func notFoundHandler(response_writer http.ResponseWriter, request *http.Request) {
	http.Error(response_writer, "Resource not found", http.StatusNotFound)
}

func methodNotAllowedHandler(response_writer http.ResponseWriter, request *http.Request) {
	http.Error(response_writer, "Method not allowed", http.StatusMethodNotAllowed)
}

func registerRoutes(router *mux.Router) {
	router.HandleFunc("/version", versionHandler).Methods("GET")
	router.HandleFunc("/get_game_information", getGameInformationHandler).Methods("GET")
	router.HandleFunc("/add_interaction", addInteractionHandler).Methods("POST")
	router.HandleFunc("/init_user", initUserHandler).Methods("GET")
	router.HandleFunc("/error404", notFoundHandler).Methods("GET")
	router.HandleFunc("/error405", methodNotAllowedHandler).Methods("POST")
}
