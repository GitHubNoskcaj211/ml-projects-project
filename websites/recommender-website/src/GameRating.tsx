import React, { useState, useEffect, useCallback } from "react";
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
  setCurrentView: React.Dispatch<
    React.SetStateAction<"LandingPage" | "FindNewGames" | "LikedGames">
  >;
}

const GameRating: React.FC<GameRatingProps> = ({ details, setCurrentView }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [history, setHistory] = useState<number[]>([]);
  const [finalGames, setFinalGames] = useState<Game[]>([]);
  const [showPopup, setShowPopup] = useState(true);
  const [startTime, setStartTime] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [allGameInfos, setAllGameInfos] = useState<Array<any>>([]);
  const [recommendations, setRecommendations] =
    useState<Recommendations | null>(null);
  const [refetch, setRefetch] = useState(false);

  const runGamesProcess = useCallback(async () => {
    console.log("Effect for fetchGameRecommendations running", details.userID);
    try {
      const new_recs = await fetchGameRecommendations();
      setRecommendations(new_recs);
      if (new_recs === null) {
        throw new Error("Error fetching game recommendations");
      }
      let gameIDs = new_recs.recommendations.map((rec) => rec.game_id);
      const fetchPromises = gameIDs.map((id) => fetchGameInfo(id));
      const gamesInfo = await Promise.all(fetchPromises);
      setAllGameInfos(gamesInfo);
      setLoading(false);
    } catch (error) {
      console.error("Error fetching games or game info:", error);
    }
  }, [details.userID]);

  useEffect(() => {
    runGamesProcess();
  }, [runGamesProcess]);

  useEffect(() => {
    if (refetch) {
      runGamesProcess();
      setRefetch(false);
    }
  }, [refetch, runGamesProcess]);

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
          // TODO: Arjun, make better
          await fetch(makeBackendURL("add_interaction"), {
            method: "POST",
            mode: "cors",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              rec_model_name: recommendations!.model_name,
              rec_model_save_path: recommendations!.model_save_path,
              num_game_interactions_local: recommendations!.num_game_interactions_local,
              num_game_owned_local: recommendations!.num_game_owned_local,
              num_game_interactions_external: recommendations!.num_game_interactions_external,
              num_game_owned_external: recommendations!.num_game_owned_external,
              game_id: allGameInfos[currentIndex].id,
              user_liked: selection,
              time_spent: timeSpentCurrent,
            }),
          });
          setFinalGames(updatedGames);
        }

        setHistory((prev) => [...prev, currentIndex]);
        setCurrentIndex(currentIndex + 1);
        setStartTime(Date.now());
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

      {currentIndex < 10 ? (
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
              <RecCircle
                value={allGameInfos[currentIndex].avgReviewScore || 0}
              />
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
      ) : (
        <div className="finalPage">
          <h1>Thank you!</h1>
          <button onClick={() => setCurrentView("LandingPage")}>
            Landing Page
          </button>
          <button
            onClick={() => {
              setRefetch(true);
              setCurrentIndex(0);
              console.log("index " + currentIndex);
              setCurrentView("FindNewGames");
            }}
          >
            Find New Games
          </button>
          <button
            onClick={() => {
              setCurrentView("LikedGames");
            }}
          >
            Liked Games
          </button>
        </div>
      )}
    </div>
  );
};

export default GameRating;
