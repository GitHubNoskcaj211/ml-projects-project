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

export async function fetchGameRecommendations(
  userId: string
): Promise<string[]> {
  try {
    const response = await axios.get<ApiResponse>(
      makeBackendURL(`/get_N_recommendations_for_user?user_id=${userId}&N=10`)
    );
    const gameIds = response.data.recommendations.map(
      (recommendation) => recommendation.game_id
    );
    console.log(gameIds);
    return gameIds;
  } catch (error) {
    console.error("Error fetching game recommendations:", error);
    return [];
  }
}
