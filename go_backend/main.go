package main

// Note: To run use `go run *.go` in go_backend folder.

import (
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/go-chi/chi/middleware"
	"github.com/go-chi/chi/v5"
	"github.com/go-chi/cors"
	"github.com/go-redis/redis/v8"
	"github.com/joho/godotenv"
)

type Config struct {
	SteamWebAPIKey string
	FrontendURL    *url.URL
	BackendURL     *url.URL
	MLBackendURL   *url.URL
	Version        string
	Name           string
	RootFolder     string
	Port           string
}

type App struct {
	Config Config
}

var app App

var redis_client *redis.Client

func main() {
	rand.Seed(time.Now().UnixNano())
	godotenv.Load()

	app = App{}
	frontendURL, err := url.Parse(getEnv("FRONTEND_URL", ""))
	if err != nil {
		log.Fatal("Failed to parse FRONTEND_URL: ", err)
	}
	backendURL, err := url.Parse(getEnv("BACKEND_URL", ""))
	if err != nil {
		log.Fatal("Failed to parse BACKEND_URL: ", err)
	}
	MLbackendURL, err := url.Parse(getEnv("ML_BACKEND_URL", ""))
	if err != nil {
		log.Fatal("Failed to parse ML_BACKEND_URL: ", err)
	}
	app.Config = Config{
		SteamWebAPIKey: getEnv("STEAM_WEB_API_KEY", ""),
		FrontendURL:    frontendURL,
		BackendURL:     backendURL,
		MLBackendURL:   MLbackendURL,
		Version:        getEnv("VERSION", ""),
		Name:           getEnv("NAME", ""),
		RootFolder:     getEnv("ROOT_FOLDER", ""),
		Port:           getEnv("PORT", "3000"),
	}

	r := chi.NewRouter()
	r.Use(beforeRequest)
	r.Use(middleware.Logger)

	frontendString := strings.TrimSuffix(frontendURL.String(), "/")
	fmt.Println("Frontend URL: ", frontendString)
	r.Use(cors.Handler(cors.Options{
		AllowedOrigins: []string{frontendString},
		AllowedHeaders: []string{"Authorization"},
	}))

	registerRoutes(r)

	log.Printf("Starting server on :%s...\n", app.Config.Port)
	log.Fatal(http.ListenAndServe(":"+app.Config.Port, r))
}

func versionHandler(w http.ResponseWriter, r *http.Request) {
	responseData := map[string]interface{}{
		"success": true,
	}
	writeJSONResponse(w, appendRequestMetaData(responseData, r))
}

func registerRoutes(r *chi.Mux) {
	r.Get("/version", versionHandler)

	r.Get("/get_game_information", getGameInformationHandler)

	r.Post("/add_interaction", requireLogin(addInteractionHandler))

	r.Get("/get_N_recommendations_for_user", requireLogin(getRecommendationsForUser))
	r.Post("/check_refresh_all_recommendation_queues", requireLogin(checkRefreshAllRecommendationQueues))

	r.Post("/init_user", requireLogin(initUserHandler))
	r.Get("/login", loginHandler)
	r.Get("/verify_login", verifyLoginHandler)
}
