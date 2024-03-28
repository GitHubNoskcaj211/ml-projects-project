import axios from "axios";
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
    const response = await axios.get(
      makeBackendURL(`get_game_information?game_id=${gameID}`)
    );
    const data = response.data;
    return {
      avgReviewScore: data.avgReviewScore,
      description: data.description,
      genres: data.genres,
      tags: data.tags,
      name: data.name,
      numFollowers: data.numFollowers,
      numReviews: data.numReviews,
      price: data.price,
      id: data.id,
    };
  } catch (error) {
    console.error("There was a problem with the Axios operation:", error);
    throw error;
  }
}
