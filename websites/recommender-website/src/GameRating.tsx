import React, { useState, useEffect } from "react";
import "./GameRating.css"; // Import the CSS file
import RecCircle from "./components/RecCircle";
import PopUpBox from "./components/PopUpBox";
import { fetchGameRecommendations } from "./components/GetRecs";
import { fetchGameInfo } from "./components/GetGameDetails";

interface Game {
  userID: string;
  gameID?: string;
  userSelection?: number;
  timeSpent?: number;
}

interface GameRatingProps {
  details: Game;
}

const GameRating: React.FC<GameRatingProps> = ({ details }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [history, setHistory] = useState<number[]>([]);
  const [finalGames, setFinalGames] = useState<Game[]>([]);
  const [showPopup, setShowPopup] = useState(true);
  const [startTime, setStartTime] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [allGameInfos, setAllGameInfos] = useState<Array<any>>([]);
  let gameIDs: Array<string> = [];

  useEffect(() => {
    console.log("Effect for fetchGameRecommendations running", details.userID);
    async function runGamesProcess() {
      try {
        gameIDs = await fetchGameRecommendations();
        const fetchPromises = gameIDs.map((id) => fetchGameInfo(id));
        const gamesInfo = await Promise.all(fetchPromises);
        setAllGameInfos(gamesInfo);
        setLoading(false);
      } catch (error) {
        console.error("Error fetching games or game info:", error);
      }
    }

    runGamesProcess();
  }, [details.userID]);

  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (loading) return;

      if (
        startTime &&
        (event.key === "ArrowRight" || event.key === "ArrowLeft")
      ) {
        const selection = event.key === "ArrowRight" ? 1 : 0;
        const updatedGames = [...finalGames];
        const timeSpentCurrent = (Date.now() - startTime) / 1000;

        if (currentIndex < allGameInfos.length) {
          updatedGames[currentIndex] = {
            ...updatedGames[currentIndex],
            gameID: allGameInfos[currentIndex].id,
            userSelection: selection,
            timeSpent: timeSpentCurrent,
          };
          setFinalGames(updatedGames);
        }

        setHistory((prev) => [...prev, currentIndex]);
        setCurrentIndex(currentIndex + 1);
        setStartTime(Date.now());
      }
      else if (event.key === "Escape") {
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
          <PopUpBox
            isOpen={showPopup}
            onClose={closePopup}
          />

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
          <pre>{JSON.stringify(finalGames, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default GameRating;
