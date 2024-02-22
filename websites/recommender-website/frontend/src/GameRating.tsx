import React, { useState, useEffect } from "react";

interface Game {
  name: string;
  rating: number;
  userSelection?: number;
}

interface GameRatingProps {
  games: Game[];
}

const GameRating: React.FC<GameRatingProps> = ({ games }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [history, setHistory] = useState<number[]>([]);
  const [finalGames, setFinalGames] = useState<Game[]>([]);

  useEffect(() => {
    setFinalGames(games.map((game) => ({ ...game })));
  }, [games]);

  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (
        (event.key === "ArrowRight" || event.key === "ArrowLeft") &&
        currentIndex <= games.length
      ) {
        const selection = event.key === "ArrowRight" ? 1 : 0;
        const updatedGames = [...finalGames];
        if (currentIndex < games.length) {
          updatedGames[currentIndex] = {
            ...updatedGames[currentIndex],
            userSelection: selection,
          };
          setFinalGames(updatedGames);
        }

        setHistory((prev) => [...prev, currentIndex]);

        if (currentIndex < games.length) {
          setCurrentIndex(currentIndex + 1);
        }
      }
    };

    window.addEventListener("keydown", handleKeyPress);

    return () => {
      window.removeEventListener("keydown", handleKeyPress);
    };
  }, [currentIndex, finalGames, games.length]);

  const handleUndo = () => {
    if (history.length > 0) {
      const previousIndex = history[history.length - 1];
      const updatedHistory = history.slice(0, -1);
      setHistory(updatedHistory);
      setCurrentIndex(previousIndex);
    }
  };

  return (
    <div>
      {currentIndex <= games.length ? (
        currentIndex < games.length ? (
          <div>
            <h1>{finalGames[currentIndex]?.name}</h1>
            <p>Rating: {finalGames[currentIndex]?.rating}</p>
          </div>
        ) : (
          <>
            <h1>Thank you!</h1>
            <pre>{JSON.stringify(finalGames, null, 2)}</pre>
          </>
        )
      ) : null}
      {history.length > 0 && <button onClick={handleUndo}>Undo</button>}
    </div>
  );
};

export default GameRating;
