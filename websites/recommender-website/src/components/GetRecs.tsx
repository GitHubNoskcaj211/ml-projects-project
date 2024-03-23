import axios from "axios";
import { makeBackendURL } from "../util";

interface ApiResponse {
  model_name: string;
  model_save_path: string;
  recommendations: Array<{ game_id: string; recommendation_score: number }>;
  time_request: number;
  execution_time_ms: number;
  version: string;
}

export async function fetchGameRecommendations(): Promise<string[]> {
  try {
    const response = await axios.get<ApiResponse>(
      makeBackendURL(`get_N_recommendations_for_user?N=10`),
      {
        withCredentials: true,
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
    return gameIds;
  } catch (error) {
    console.error("Error fetching game recommendations:", error);
    return [];
  }
}
