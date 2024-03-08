import React, { useEffect, useState } from "react";
import GameRating from "./GameRating";

function makeBackendURL(path: string) {
  const url = new URL(path, import.meta.env.VITE_BACKEND_URL);
  return url.toString();
}

const App: React.FC = () => {
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
        <button onClick={() => (location.href = makeBackendURL("/login"))}>
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
