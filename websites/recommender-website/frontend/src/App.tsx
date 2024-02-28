import React from "react";
import GameRating from "./GameRating";

const App: React.FC = () => {
  const gameRatings = [
    { name: "Baldur's Gate 3", rating: 0.97 },
    { name: "CSGO 2", rating: 0.9 },
    { name: "Speedrunners", rating: 0.5 },
  ];

  return (
    <div>
      <GameRating games={gameRatings} />
    </div>
  );
};

export default App;
