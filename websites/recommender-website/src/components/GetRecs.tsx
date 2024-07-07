import { backendAuthFetch } from "../util";

export interface RecResponse {
  avgReviewScore: number;
  description: string;
  genres: string[];
  name: string;
  numFollowers: number;
  numReviews: number;
  price: number;
  tags: string[];
  game_id: string;
  recommendation_score: number;
  model_name: string;
  model_save_path: string;
  num_game_interactions_local: number;
  num_game_owned_local: number;
  num_game_interactions_external: number;
  num_game_owned_external: number;
}

export interface RecommendationsResponse {
  recommendations: Array<RecResponse>;
}

export async function fetchGameRecommendations(
  num_games: number,
  exclude_game_ids: string[]
): Promise<RecommendationsResponse> {
  const exclude_game_ids_string = exclude_game_ids.length == 0 ? "" : "&exclude_game_id=" + exclude_game_ids.join("&exclude_game_id=") 
  const resp = await backendAuthFetch(
    `get_N_recommendations_for_user?N=${num_games}${exclude_game_ids_string}`
  );
  return resp.json();
}
