import axios from "axios";
import { makeBackendURL } from "../util";

export interface Recommendations {
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
): Promise<Recommendations | null> {
  try {
    const response = await axios.get<Recommendations>(
      makeBackendURL(`get_N_recommendations_for_user?N=${num_games}`),
      {
        signal,
      }
    );
    const gameIds = response.data.recommendations.map(
      (recommendation) => recommendation.game_id
    );
    console.log(`model_name: ${response.data.model_name}`);
    console.log(`model_save_path: ${response.data.model_save_path}`);
    console.log(`time_request: ${response.data.time_request}`);
    console.log(`execution_time_ms: ${response.data.execution_time_ms}`);
    console.log(`version: ${response.data.version}`);
    console.log(gameIds);
    return response.data;
  } catch (error) {
    console.error("Error fetching game recommendations:", error);
    return null;
  }
}
