import axios from "axios";
import { makeBackendURL } from "../util";

export interface RecommendationsResponse {
  recommendations: Array<{ game_id: number; recommendation_score: number }>;
  model_name: string;
  model_save_path: string;
  num_game_interactions_local: number;
  num_game_owned_local: number;
  num_game_interactions_external: number;
  num_game_owned_external: number;
  time_request: number;
  execution_time_ms: number;
  version: string;
}

export async function fetchGameRecommendations(
  num_games: number,
  signal: AbortSignal
): Promise<RecommendationsResponse> {
  const resp = await fetch(makeBackendURL(`get_N_recommendations_for_user?N=${num_games}`), {
    signal,
    credentials: "include",
  });
  return resp.json();
}
