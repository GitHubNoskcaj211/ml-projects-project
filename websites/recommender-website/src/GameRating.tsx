import React, { useState, useEffect } from "react";
import "./GameRating.css";
import RecCircle from "./components/RecCircle";
import PopUpBox from "./components/PopUpBox";
import {
  fetchGameRecommendations,
  Recommendations,
} from "./components/GetRecs";
import { fetchGameInfo } from "./components/GetGameDetails";
import { makeBackendURL } from "./util";

interface Game {
  userID: string;
  gameID?: string;
  userSelection?: boolean;
  timeSpent?: number;
}

interface GameRatingProps {
  details: Game;
}

const BATCH_SIZE = 5;

const GameRating: React.FC<GameRatingProps> = ({ details }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [history, setHistory] = useState<number[]>([]);
  const [finalGames, setFinalGames] = useState<Game[]>([]);
  const [showPopup, setShowPopup] = useState(true);
  const [startTime, setStartTime] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [allGameInfos, setAllGameInfos] = useState<Array<any>>([]);
  const [recommendations, setRecommendations] =
    useState<Recommendations | null>(null);

  const runGamesProcess = async (signal: AbortSignal) => {
    console.log("Effect for fetchGameRecommendations running", details.userID);
    try {
      const new_recs = await fetchGameRecommendations(BATCH_SIZE, signal);
      setRecommendations(new_recs);
      if (new_recs === null) {
        throw new Error("Error fetching game recommendations");
      }
      let gameIDs = new_recs.recommendations.map((rec) => rec.game_id);
      const fetchPromises = gameIDs.map((id) => fetchGameInfo(id));
      const gamesInfo = await Promise.all(fetchPromises);
      setCurrentIndex(0);
      setAllGameInfos(gamesInfo);
      setLoading(false);
      setStartTime(Date.now());
    } catch (error) {
      console.error("Error fetching games or game info:", error);
    }
  };

  useEffect(() => {
    if (loading) {
      const controller = new AbortController();
      runGamesProcess(controller.signal);
      return () => {
        controller.abort();
      };
    }
  }, [loading]);

  useEffect(() => {
    const handleKeyPress = async (event: KeyboardEvent) => {
      if (loading) return;

      if (
        startTime &&
        (event.key === "ArrowRight" || event.key === "ArrowLeft")
      ) {
        const selection = event.key === "ArrowRight";
        const updatedGames = [...finalGames];
        const timeSpentCurrent = (Date.now() - startTime) / 1000;

        if (currentIndex < allGameInfos.length) {
          updatedGames[currentIndex] = {
            ...updatedGames[currentIndex],
            gameID: allGameInfos[currentIndex].id,
            userSelection: selection,
            timeSpent: timeSpentCurrent,
          };
          await fetch(makeBackendURL("add_interaction"), {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              rec_model_name: recommendations!.model_name,
              rec_model_save_path: recommendations!.model_save_path,
              num_game_interactions_local:
                recommendations!.num_game_interactions_local,
              num_game_owned_local: recommendations!.num_game_owned_local,
              num_game_interactions_external:
                recommendations!.num_game_interactions_external,
              num_game_owned_external: recommendations!.num_game_owned_external,
              game_id: allGameInfos[currentIndex].id,
              user_liked: selection,
              time_spent: timeSpentCurrent,
            }),
          });
          setFinalGames(updatedGames);
        }

        if (currentIndex === allGameInfos.length - 1) {
          setLoading(true);
        } else {
          setHistory((prev) => [...prev, currentIndex]);
          setCurrentIndex(currentIndex + 1);
          setStartTime(Date.now());
        }
      } else if (event.key === "Escape") {
        closePopup();
      }
    };

    window.addEventListener("keydown", handleKeyPress);
    return () => {
      window.removeEventListener("keydown", handleKeyPress);
    };
  }, [currentIndex, finalGames, allGameInfos.length, startTime, loading]);

  const handleUndo = () => {
    if (history.length > 0 && !loading) {
      const previousIndex = history.pop()!;
      setHistory([...history]);
      setCurrentIndex(previousIndex);
    }
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
          <h1>{allGameInfos[currentIndex].name}</h1>
        </div>
        <div className="secondRow">
          {/* Image */}
          <div className="image box">
            <img
              src={`https://cdn.akamai.steamstatic.com/steam/apps/${allGameInfos[currentIndex].id}/header.jpg`}
              alt={allGameInfos[currentIndex].name}
            />
          </div>

          {/* RecCircle */}
          <div className="rec box">
            <RecCircle value={allGameInfos[currentIndex].avgReviewScore || 0} />
          </div>
        </div>

        {/* Game Description */}
        <div className="game box">
          <p>{allGameInfos[currentIndex].description}</p>
        </div>

        {/* Genres */}
        <div className="genre box">
          <h2>Genres</h2>
          <div className="genreButtons">
            {allGameInfos[currentIndex].genres.map(
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
            {allGameInfos[currentIndex].tags.map(
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
          {history.length > 0 && (
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
