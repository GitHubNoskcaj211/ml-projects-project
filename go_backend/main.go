package main

// Note: To run use `go run *.go` in go_backend folder.

import (
	"log"
	"net/http"
	"strings"

	"github.com/go-chi/chi/middleware"
	"github.com/go-chi/chi/v5"
	"github.com/go-chi/cors"
	"github.com/joho/godotenv"
)

type Config struct {
	SteamWebAPIKey               string
	FrontendURL                  string
	BackendURL                   string
	GoogleApplicationCredentials string
	Version                      string
	Name                         string
	RootFolder                   string
	Port                         string
}

type App struct {
	Config Config
}

var app App

func main() {
	godotenv.Load()

	app = App{}
	app.Config = Config{
		SteamWebAPIKey:               getEnv("STEAM_WEB_API_KEY", ""),
		FrontendURL:                  getEnv("FRONTEND_URL", ""),
		BackendURL:                   getEnv("BACKEND_URL", ""),
		GoogleApplicationCredentials: getEnv("GOOGLE_APPLICATION_CREDENTIALS", ""),
		Version:                      getEnv("VERSION", ""),
		Name:                         getEnv("NAME", ""),
		RootFolder:                   getEnv("ROOT_FOLDER", ""),
		Port:                         getEnv("PORT", "3000"),
	}

	r := chi.NewRouter()
	r.Use(beforeRequest)
	r.Use(middleware.Logger)

	frontendURL := strings.TrimRight(app.Config.FrontendURL, "/")
	r.Use(cors.Handler(cors.Options{
		AllowedOrigins: []string{frontendURL},
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

	r.Post("/init_user", requireLogin(initUserHandler))
	r.Get("/login", loginHandler)
	r.Get("/verify_login", verifyLoginHandler)
}
