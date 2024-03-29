import React, { useState, useEffect } from "react";
import "./GameRating.css";
import RecCircle from "./components/RecCircle";
import PopUpBox from "./components/PopUpBox";
import {
  fetchGameRecommendations,
  RecommendationsResponse,
} from "./components/GetRecs";
import { fetchGameInfo, GameInfo } from "./components/GetGameDetails";
import { makeBackendURL } from "./util";

interface Game {
  userID: string;
  gameID?: string;
  userSelection?: boolean;
  timeSpent?: number;
}

interface Recommendation {
  gameInfo: GameInfo;
  resp: RecommendationsResponse;
}

interface GameRatingProps {
  details: Game;
}

const BATCH_SIZE = 5;
const HISTORY_MAX_SIZE = 100;

const GameRating: React.FC<GameRatingProps> = ({ details }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [showPopup, setShowPopup] = useState(true);
  const [startTime, setStartTime] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [fetchingGames, setFetchingGames] = useState(false);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);

  const runGamesProcess = async (signal: AbortSignal) => {
    console.log("Effect for fetchGameRecommendations running", details.userID);
    try {
      const resp = await fetchGameRecommendations(BATCH_SIZE, signal);
      const promises = resp.recommendations.map(async (rec) => {
        const gameInfo = await fetchGameInfo(rec.game_id);
        return {
          gameInfo,
          resp,
        };
      });
      const recommendations = await Promise.all(promises);
      setRecommendations((prev) => {
        const prevRecommend = new Set(prev.map((rec) => rec.gameInfo.id));
        let addedRecs = recommendations.filter(
          (game) => !prevRecommend.has(game.gameInfo.id)
        );
        let result = prev.concat(addedRecs);
        if (result.length > HISTORY_MAX_SIZE) {
          const numExtra = result.length - HISTORY_MAX_SIZE;
          result = result.slice(-numExtra);
          setCurrentIndex((prev) => prev - numExtra);
        }
        return result;
      });
      setFetchingGames(false);
      setLoading(false);
    } catch (error) {
      console.error("Error fetching games or game info:", error);
    }
  };

  useEffect(() => {
    if (!fetchingGames) {
      return;
    }
    const controller = new AbortController();
    runGamesProcess(controller.signal);
    return () => {
      controller.abort();
    };
  }, [fetchingGames]);

  useEffect(() => {
    if (showPopup) {
      const handleKeyPress = (event: KeyboardEvent) => {
        if (event.key === "Escape") {
          closePopup();
        }
      }
      window.addEventListener("keydown", handleKeyPress);
      return () => {
        window.removeEventListener("keydown", handleKeyPress);
      };
    }

    const handleKeyPress = (event: KeyboardEvent) => {
      if (loading) return;

      if (startTime === null) {
        throw Error("startTime is null");
      }
      if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") {
        return;
      }
      const userLiked = event.key === "ArrowRight";
      const timeSpent = (Date.now() - startTime) / 1000;

      console.assert(currentIndex < recommendations.length);
      const rec = recommendations[currentIndex];
      // No await to make user experience faster
      fetch(makeBackendURL("add_interaction"), {
        credentials: "include",
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          rec_model_name: rec.resp.model_name,
          rec_model_save_path: rec.resp.model_save_path,
          num_game_interactions_local: rec.resp.num_game_interactions_local,
          num_game_owned_local: rec.resp.num_game_owned_local,
          num_game_interactions_external:
            rec.resp.num_game_interactions_external,
          num_game_owned_external: rec.resp.num_game_owned_external,
          game_id: rec.gameInfo.id,
          user_liked: userLiked,
          time_spent: timeSpent,
        }),
      });
      const newIndex = currentIndex + 1;
      if (newIndex >= recommendations.length) {
        setLoading(true);
      }
      setCurrentIndex(newIndex);
      setStartTime(Date.now());
    };

    window.addEventListener("keydown", handleKeyPress);
    return () => {
      window.removeEventListener("keydown", handleKeyPress);
    };
  }, [showPopup, loading, startTime, currentIndex, recommendations.length]);

  useEffect(() => {
    const numLeft = recommendations.length - currentIndex;
    if (numLeft <= BATCH_SIZE) {
      setFetchingGames(true);
    }
    if (numLeft === 0) {
      setStartTime(null);
    } else {
      setStartTime((prev) => (prev ? prev : Date.now()));
    }
  }, [currentIndex, recommendations]);

  const handleUndo = () => {
    console.assert(currentIndex > 0);
    setCurrentIndex((prev) => prev - 1);
    setStartTime(Date.now());
  };

  const closePopup = () => {
    setShowPopup(false);
    setStartTime(Date.now());
  };

  if (loading) {
    return <div>Loading...</div>;
  }

  return (
    <div className="container">
      {
        /* Popup Directions Box*/
        showPopup && currentIndex === 0 && (
          <PopUpBox isOpen={showPopup} onClose={closePopup} />
        )
      }

      <div className="contentContainer">
        {/* Game Title */}
        <div className="title box">
          <h1>{recommendations[currentIndex].gameInfo.name}</h1>
        </div>
        <div className="secondRow">
          {/* Image */}
          <div className="image box">
            <img
              src={`https://cdn.akamai.steamstatic.com/steam/apps/${recommendations[currentIndex].gameInfo.id}/header.jpg`}
              alt={recommendations[currentIndex].gameInfo.name}
            />
          </div>

          {/* RecCircle */}
          <div className="rec box">
            <RecCircle
              value={recommendations[currentIndex].gameInfo.avgReviewScore || 0}
            />
          </div>
        </div>

        {/* Game Description */}
        <div className="game box">
          <p>{recommendations[currentIndex].gameInfo.description}</p>
        </div>

        {/* Genres */}
        <div className="genre box">
          <h2>Genres</h2>
          <div className="genreButtons">
            {recommendations[currentIndex].gameInfo.genres.map(
              (genre: string, index: number) => (
                <button key={index} disabled>
                  {genre}
                </button>
              )
            )}
          </div>
        </div>

        {/* Tags */}
        <div className="tag box">
          <h2>Tags</h2>
          <div className="tagButtons">
            {recommendations[currentIndex].gameInfo.tags.map(
              (tag: string, index: number) => (
                <button key={index} disabled>
                  {tag}
                </button>
              )
            )}
          </div>
        </div>

        {/* Undo Button */}
        <div className="undoContainer">
          {currentIndex > 0 && (
            <button className="undoButton" onClick={handleUndo}>
              Go Back
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default GameRating;
