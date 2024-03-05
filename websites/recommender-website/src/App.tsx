import React, { useEffect, useState } from "react";
import GameRating from "./GameRating";

function makeBackendURL(path: string) {
  return `${import.meta.env.VITE_BACKEND_URL}/${path}`
}

const App: React.FC = () => {
  const gameRatings = [
    { name: "Baldur's Gate 3", rating: 0.97 },
    { name: "CSGO 2", rating: 0.9 },
    { name: "Speedrunners", rating: 0.5 },
  ];

  const [userID, setUserID] = useState<string | undefined | null>(undefined);

  useEffect(() => {
    (async () => {
      const res = await fetch(makeBackendURL("/user"), {
        mode: "cors",
        credentials: "include",
      });
      if (res.status == 401) {
        setUserID(null);
        return;
      }
      const data = await res.json();
      setUserID(data.id);
    })();
  }, []);

  if (userID === undefined) {
    return <div className="container">Loading...</div>;
  }

  if (userID === null) {
    return (
      <div className="container signInContainer">
        <button onClick={() => location.href = "http://127.0.0.1:3000/login"}>Sign in through Steam</button>
      </div>
    );
  }

  return (
    <div>
      <GameRating games={gameRatings} />
    </div>
  );
};

export default App;
