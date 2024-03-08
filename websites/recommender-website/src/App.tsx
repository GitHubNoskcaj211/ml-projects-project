import React, { useEffect, useState } from "react";
import GameRating from "./GameRating";

const App: React.FC = () => {
  const [userID, setUserID] = useState<string | undefined | null>(undefined);

  useEffect(() => {
    (async () => {
      const res = await fetch("/api/user");
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
        <button onClick={() => (location.href = "/api/login")}>
          Sign in through Steam
        </button>
      </div>
    );
  }

  return (
    <div>
      {userID !== undefined && userID !== null && (
        <GameRating details={{ userID }} />
      )}
    </div>
  );
};

export default App;
