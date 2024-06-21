import { makeBackendURL } from "../util";

export interface GameInfo {
  avgReviewScore: number;
  description: string;
  genres: string[];
  name: string;
  numFollowers: number;
  numReviews: number;
  price: number;
  tags: string[];
  id: string;
}

export async function fetchGameInfo(gameID: number): Promise<GameInfo> {
  try {
    const response = await fetch(
      makeBackendURL(`get_game_information?game_id=${gameID}`)
    );
    return await response.json();
  } catch (error) {
    console.error("There was a problem with the Axios operation:", error);
    throw error;
  }
}
