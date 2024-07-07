package main

import (
	"context"
	"fmt"
	"math/rand"
	"net/http"
	"strconv"

	firestorepb "cloud.google.com/go/firestore/apiv1/firestorepb"
	"google.golang.org/api/iterator"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const NUM_RECOMMENDATIONS_IN_QUEUE = 120
const NUM_INTERACTIONS_REFRESH_QUEUE = 30

func makeRecommendationMap(response_writer http.ResponseWriter, metadata_for_model map[string]interface{}, recommendation map[string]interface{}) map[string]interface{} {
	game_information := getGameInformation(response_writer, strconv.FormatInt(recommendation["game_id"].(int64), 10))
	if game_information == nil {
		return nil
	}
	return mergeMaps(metadata_for_model, recommendation, game_information)
}

func removeIndexFromList(lst []string, index int) []string {
	lst[index] = lst[len(lst)-1]
	return lst[:len(lst)-1]
}

func getRandomlySelectedRecommendations(response_writer http.ResponseWriter, userID int64, num_recommendations_requested int64, exclude_game_ids_set map[int64]struct{}) []map[string]interface{} {
	firestoreClient := getFirestoreClient()
	recommendation_queue := firestoreClient.Collection("recommendation_queue").Doc(strconv.FormatInt(userID, 10))
	current_recommendation_index_for_model := make(map[string]int)
	models_to_select_from := make([]string, 0)
	recommendation_queue_for_model := make(map[string][]map[string]interface{})
	recommendation_queue_metadata_for_model := make(map[string]map[string]interface{})

	collections, err := recommendation_queue.Collections(context.Background()).GetAll()
	if err != nil {
		writeErrorJSONResponse(response_writer, "Error receiving recommendation queue collections for user.", http.StatusInternalServerError)
		return nil
	}

	for _, collection := range collections {
		collection_name := collection.ID
		models_to_select_from = append(models_to_select_from, collection_name)
		current_recommendation_index_for_model[collection_name] = 0

		recommendation_queue_doc := collection.Doc("recommendation_queue")
		recommendation_queue_snap, err := recommendation_queue_doc.Get(context.Background())
		if err != nil {
			writeErrorJSONResponse(response_writer, "Error receiving recommendation queue document for collection "+collection_name, http.StatusInternalServerError)
			return nil
		}
		recommendations, err := recommendation_queue_snap.DataAt("recommendations")
		if err != nil {
			writeErrorJSONResponse(response_writer, "Error receiving recommendations for collection "+collection_name, http.StatusInternalServerError)
			return nil
		}
		if recommendations_array, ok := recommendations.([]interface{}); ok {
			for _, recommendation := range recommendations_array {
				if recommendation_map, ok := recommendation.(map[string]interface{}); ok {
					recommendation_queue_for_model[collection_name] = append(recommendation_queue_for_model[collection_name], recommendation_map)
				}
			}
		}

		recommendation_metadata_doc := collection.Doc("recommendation_metadata")
		recommendation_metadata_snap, err := recommendation_metadata_doc.Get(context.Background())
		if err != nil {
			writeErrorJSONResponse(response_writer, "Error receiving recommendation metadata document for collection "+collection_name, http.StatusInternalServerError)
			return nil
		}
		recommendation_queue_metadata_for_model[collection_name] = recommendation_metadata_snap.Data()
	}

	recommendations := make([]map[string]interface{}, 0)
	for len(models_to_select_from) > 0 && len(recommendations) < int(num_recommendations_requested) {
		random_model_index := rand.Intn(len(models_to_select_from))
		random_model := models_to_select_from[random_model_index]

		recommendation_index := current_recommendation_index_for_model[random_model]
		game_id := recommendation_queue_for_model[random_model][recommendation_index]["game_id"].(int64)
		if _, found := exclude_game_ids_set[game_id]; !found {
			recommendation_map := makeRecommendationMap(response_writer, recommendation_queue_metadata_for_model[random_model], recommendation_queue_for_model[random_model][recommendation_index])
			if recommendation_map == nil {
				return nil
			}
			recommendations = append(recommendations, recommendation_map)
			exclude_game_ids_set[game_id] = struct{}{}
		}
		current_recommendation_index_for_model[random_model]++
		if current_recommendation_index_for_model[random_model] >= len(recommendation_queue_for_model[random_model]) {
			models_to_select_from = removeIndexFromList(models_to_select_from, random_model_index)
		}
	}
	return recommendations
}

func getRecommendationsForUser(response_writer http.ResponseWriter, request *http.Request, userID int64) {
	url_values := request.URL.Query()
	num_recommendations_requested := validateParameterInt("N", response_writer, url_values, true)
	exclude_game_ids := validateParameterIntList("exclude_game_ids", response_writer, url_values, false)
	if num_recommendations_requested == nil {
		return
	}
	if exclude_game_ids == nil {
		exclude_game_ids = make([]int64, 0)
	}

	exclude_game_ids_set := make(map[int64]struct{})
	for _, elem := range exclude_game_ids {
		exclude_game_ids_set[elem] = struct{}{}
	}

	firestoreClient := getFirestoreClient()
	interactedGamesCollection := firestoreClient.Collection("interactions").Doc("data").Collection(strconv.FormatInt(userID, 10))
	iter := interactedGamesCollection.Documents(context.Background())
	defer iter.Stop()
	for {
		doc, err := iter.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			writeErrorJSONResponse(response_writer, "Error iterating over interacted games document.", http.StatusInternalServerError)
			return
		}
		gameID, err := doc.DataAt("game_id")
		if err != nil {
			writeErrorJSONResponse(response_writer, "Error getting game_id from interaction documnt.", http.StatusInternalServerError)
			return
		}
		exclude_game_ids_set[gameID.(int64)] = struct{}{}
	}

	randomly_selected_recommendations := getRandomlySelectedRecommendations(response_writer, userID, num_recommendations_requested.(int64), exclude_game_ids_set)
	if randomly_selected_recommendations == nil {
		return
	}

	response := map[string]interface{}{
		"recommendations": randomly_selected_recommendations,
	}

	// TODO need a backup in case we send back nothing
	writeJSONResponse(response_writer, appendRequestMetaData(response, request))
}

var model_save_paths = []string{
	"evaluation_test_common_neighbors_constant_scoring",
	"evaluation_test_common_neighbors_percentile_scoring",
	"evaluation_test_common_neighbors",
	"evaluation_test_popularity_model",
	"evaluation_test_cf_low_weight_decay_increased_lr_best_model_bugfix",
	"evaluation_test_gcf_low_weight_decay_increased_lr_best_model_bugfix",
	"evaluation_test_mlp_low_weight_decay_increased_lr_best_model_bugfix",
	"evaluation_test_ncf_low_weight_decay_increased_lr_best_model_bugfix",
	"evaluation_test_cf_embed_all_except_tags_genres_best_model_bugfix_clip_embeddings",
	"evaluation_test_gcf_embed_all_except_tags_genres_best_model_bugfix_clip_embeddings",
	"evaluation_test_mlp_embed_all_except_tags_genres_best_model_bugfix_clip_embeddings",
	"evaluation_test_ncf_embed_all_except_tags_genres_best_model_bugfix_clip_embeddings",
}

func shouldRefreshRecommendationQueue(userID int64, model_save_path string, num_user_interactions int64) interface{} {
	lock_key := fmt.Sprintf("refreshing:%d:%s", userID, model_save_path)
	firestoreClient := getFirestoreClient()
	docRef := firestoreClient.Collection("locks").Doc(lock_key)
	_, err := docRef.Get(context.Background())
	if err == nil || status.Code(err) != codes.NotFound {
		return false
	}

	recommendation_metadata_snap, err := firestoreClient.Collection("recommendation_queue").Doc(strconv.FormatInt(userID, 10)).Collection(model_save_path).Doc("recommendation_metadata").Get(context.Background())
	if err != nil {
		return true
	}
	recommendation_queue_snap, err := firestoreClient.Collection("recommendation_queue").Doc(strconv.FormatInt(userID, 10)).Collection(model_save_path).Doc("recommendation_queue").Get(context.Background())
	if err != nil {
		return true
	}
	recommendation_queue := recommendation_queue_snap.Data()
	if len(recommendation_queue["recommendations"].([]interface{})) != NUM_RECOMMENDATIONS_IN_QUEUE {
		return true
	}

	recommendation_queue_metadata_for_model := recommendation_metadata_snap.Data()
	return num_user_interactions >= recommendation_queue_metadata_for_model["num_game_interactions_external"].(int64)+NUM_INTERACTIONS_REFRESH_QUEUE
}

func checkRefreshAllRecommendationQueues(response_writer http.ResponseWriter, request *http.Request, userID int64) {
	lock_key := fmt.Sprintf("check_refreshing:%d", userID)
	if err := acquireLock(lock_key); err != nil {
		writeErrorJSONResponse(response_writer, "Another process is currently calling route check_refreshing for this user. Please try again later. Error: "+fmt.Sprintf("%v", err), http.StatusConflict)
		return
	}
	defer releaseLock(lock_key)

	firestoreClient := getFirestoreClient()
	collection := firestoreClient.Collection("interactions").Doc("data").Collection(strconv.FormatInt(userID, 10))
	aggregationQuery := collection.NewAggregationQuery().WithCount("all")
	results, err := aggregationQuery.Get(context.Background())
	if err != nil {
		writeErrorJSONResponse(response_writer, "Failed to save interaction: "+err.Error(), http.StatusInternalServerError)
		return
	}

	num_user_interactions, ok := results["all"]
	if !ok {
		writeErrorJSONResponse(response_writer, "firestore: couldn't get alias for COUNT from results", http.StatusInternalServerError)
		return
	}

	for _, model_save_path := range model_save_paths {
		result := shouldRefreshRecommendationQueue(userID, model_save_path, num_user_interactions.(*firestorepb.Value).GetIntegerValue())
		if result == nil {
			return
		}
		if result.(bool) {
			makeAsyncRequest(app.Config.MLBackendURL.String() + "refresh_recommendation_queue?N=" + strconv.FormatInt(NUM_RECOMMENDATIONS_IN_QUEUE, 10) + "&model_save_file_name=" + model_save_path + "&user_id=" + strconv.FormatInt(userID, 10))
		}
	}

	response := map[string]interface{}{
		"success": 1,
	}
	writeJSONResponse(response_writer, appendRequestMetaData(response, request))
}
