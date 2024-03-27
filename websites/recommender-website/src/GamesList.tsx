import React from "react";

interface GamesListProps {
  userID: string;
  setCurrentView: React.Dispatch<
    React.SetStateAction<"LandingPage" | "FindNewGames" | "LikedGames">
  >;
}

const GamesList: React.FC<GamesListProps> = ({ userID, setCurrentView }) => {
  return (
    <div>
      <h1>Hi, User {userID}</h1>
      <button onClick={() => setCurrentView("LandingPage")}>
        Landing Page
      </button>
      <button onClick={() => setCurrentView("FindNewGames")}>
        Find New Games
      </button>
    </div>
  );
};

export default GamesList;
