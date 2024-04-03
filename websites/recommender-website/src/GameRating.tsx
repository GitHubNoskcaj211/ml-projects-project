import React, { useState, useEffect } from "react";
import "./GameRating.css";
import RecCircle from "./components/RecCircle";
import PopUpBox from "./components/PopUpBox";
import {
  fetchGameRecommendations,
  RecommendationsResponse,
  RecResponse,
} from "./components/GetRecs";
import { makeBackendURL } from "./util";

interface Game {
  userID: string;
  gameID?: string;
  userSelection?: boolean;
  timeSpent?: number;
}

interface Recommendation {
  resp: RecommendationsResponse;
  rec: RecResponse;
}

interface GameRatingProps {
  details: Game;
}

const REQ_BATCH_SIZE = 10;
const BUFFER_SIZE = 40;

const GameRating: React.FC<GameRatingProps> = ({ details }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [showPopup, setShowPopup] = useState(true);
  const [startTime, setStartTime] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [interactionAttempts, setInteractionAttempts] = useState(0);
  const [expectedRecommendationsLength, setExpectedRecommendationsLength] =
    useState(0);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [steamLinkClicked, setSteamLinkClicked] = useState(false);

  const handleSteamLinkClicked = () => {
    setSteamLinkClicked(true);
  };

  const runGamesProcess = async () => {
    console.log("Effect for fetchGameRecommendations running", details.userID);
    while (true) {
      try {
        const resp = await fetchGameRecommendations(
          REQ_BATCH_SIZE + BUFFER_SIZE
        );
        setRecommendations((prev) => {
          const prevRecommend = new Set(prev.map((rec) => rec.rec.id));
          const addedRecs = resp.recommendations.filter(
            (rec) => !prevRecommend.has(rec.id)
          )
          .slice(0, REQ_BATCH_SIZE)
          .map((rec => ({
            resp,
            rec,
          })));
          return prev.concat(addedRecs);
        });
        setLoading(false);
        return;
      } catch (e) {
        console.error("Error fetching games or game info. Trying again:", e);
      }
    }
  };

  useEffect(() => {
    if (showPopup) {
      const handleKeyPress = (event: KeyboardEvent) => {
        if (event.key === "Escape") {
          closePopup();
        }
      };
      window.addEventListener("keydown", handleKeyPress, { once: true });
      return () => {
        window.removeEventListener("keydown", handleKeyPress);
      };
    }

    if (loading || interactionAttempts > 1) {
      return;
    }
    const handleKeyPress = async (event: KeyboardEvent) => {
      if (loading) return;

      if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") {
        return;
      }
      if (startTime === null) {
        throw Error("startTime is null");
      }

      setInteractionAttempts((prev) => prev + 1);
      if (interactionAttempts > 1) {
        return;
      }
      const userLiked = event.key === "ArrowRight";
      const timeSpent = (Date.now() - startTime) / 1000;

      console.assert(currentIndex < recommendations.length);
      const rec = recommendations[currentIndex];
      const newIndex = currentIndex + 1;
      await fetch(makeBackendURL("add_interaction"), {
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
          game_id: rec.rec.id,
          user_liked: userLiked,
          time_spent: timeSpent,
          steam_link_clicked: steamLinkClicked,
        }),
      });
      if (newIndex >= recommendations.length) {
        setLoading(true);
      }
      setInteractionAttempts(0);
      setCurrentIndex(newIndex);
      setStartTime(Date.now());
      setSteamLinkClicked(false);
    };

    window.addEventListener("keydown", handleKeyPress, { once: true });
    return () => {
      window.removeEventListener("keydown", handleKeyPress);
    };
  }, [
    showPopup,
    loading,
    startTime,
    currentIndex,
    steamLinkClicked,
    interactionAttempts,
    recommendations.length,
  ]);

  useEffect(() => {
    const expectedNumLeft = expectedRecommendationsLength - currentIndex;
    if (expectedNumLeft >= BUFFER_SIZE) {
      return;
    }
    let numBatches = 0;
    numBatches = Math.ceil((BUFFER_SIZE - expectedNumLeft) / REQ_BATCH_SIZE);
    setExpectedRecommendationsLength(
      (prev) => prev + numBatches * REQ_BATCH_SIZE
    );
    for (let i = 0; i < numBatches; i++) {
      runGamesProcess();
    }
  }, [currentIndex]);

  useEffect(() => {
    if (recommendations.length - currentIndex === 0) {
      setStartTime(null);
    } else {
      setStartTime((prev) => (prev ? prev : Date.now()));
    }
  }, [currentIndex, recommendations.length]);

  const handleUndo = () => {
    console.assert(currentIndex > 0);
    setCurrentIndex((prev) => prev - 1);
    setStartTime(Date.now());
  };

  const closePopup = () => {
    setShowPopup(false);
    setStartTime(Date.now());
  };

  if (loading || interactionAttempts > 1) {
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
          <a
            href={`https://store.steampowered.com/app/${recommendations[currentIndex].rec.id}`}
            target="_blank"
            rel="noopener noreferrer"
            onClick={handleSteamLinkClicked}
          >
            <h1>{recommendations[currentIndex].rec.name}</h1>
          </a>
        </div>
        {/* Price */}
        <div className="price box">
          <h2>Price: ${recommendations[currentIndex].rec.price}</h2>
        </div>
        <div className="secondRow">
          {/* Image */}
          <div className="image box">
            <img
              src={`https://cdn.akamai.steamstatic.com/steam/apps/${recommendations[currentIndex].rec.id}/header.jpg`}
              alt={recommendations[currentIndex].rec.name}
            />
          </div>

          {/* RecCircle */}
          <div className="rec box">
            <RecCircle
              value={recommendations[currentIndex].rec.avgReviewScore || 0}
              num_reviewers={
                recommendations[currentIndex].rec.numReviews || 0
              }
            />
          </div>
        </div>

        {/* Game Description */}
        <div className="game box">
          <p>{recommendations[currentIndex].rec.description}</p>
        </div>

        {/* Genres */}
        <div className="genre box">
          <h2>Genres</h2>
          <div className="genreButtons">
            {recommendations[currentIndex].rec.genres.map(
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
            {recommendations[currentIndex].rec.tags.map(
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
