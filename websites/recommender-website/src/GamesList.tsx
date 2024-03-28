import React from "react";

interface GamesListProps {
  userID: string;
}

const GamesList: React.FC<GamesListProps> = ({ userID }) => {
  return (
    <div>
      <h1>Hi, User {userID}</h1>
    </div>
  );
};

export default GamesList;
