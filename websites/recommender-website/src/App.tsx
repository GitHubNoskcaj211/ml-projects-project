import React, { useEffect, useState } from "react";
import GameRating from "./GameRating";
import GamesList from "./GamesList";
import PublicDirectionsBox from "./components/publicDirections";
import "./App.css";

import { makeBackendURL } from "./util";

const App: React.FC = () => {
  const [userID, setUserID] = useState<string | undefined | null>(undefined);
  const [showPublicProfileWarning, setShowPublicProfileWarning] = useState(false);
  const [currentView, setCurrentView] = useState<
    "LandingPage" | "FindNewGames" | "LikedGames"
  >("LandingPage");

  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (showPublicProfileWarning && event.key === "Escape") {
        closePopup();
      }
    };

    window.addEventListener("keydown", handleKeyPress);
    return () => {
      window.removeEventListener("keydown", handleKeyPress);
    };
  }, [showPublicProfileWarning]);

  useEffect(() => {
    const controller = new AbortController();
    (async () => {
      const res = await fetch(makeBackendURL("init_user"), {
        credentials: "include",
        signal: controller.signal,
      });
      if (res.status === 401) {
        setUserID(null);
        setShowPublicProfileWarning(false);
        return;
      } else if (res.status === 500) {
        setUserID(null);
        setShowPublicProfileWarning(true);
        return;
      }
      const data = await res.json();
      setUserID(data.id);
    })();
    return () => {
      controller.abort();
    };
  }, []);

  const closePopup = () => {
    setShowPublicProfileWarning(false);
  };

  if (userID === undefined) {
    return <div className="container">Loading...</div>;
  }

  if (userID === null) {
    return (
      <div className="container signInContainer">
        {
          /* Popup Directions Box*/
          showPublicProfileWarning && (
            <PublicDirectionsBox isOpen={showPublicProfileWarning} onClose={closePopup} />
          )
        }
        <button onClick={() => (location.href = makeBackendURL("/login"))}>
          Sign in through Steam
        </button>
      </div>
    );
  }

  if (currentView === "LandingPage") {
    return (
      <div className="landingPage">
        <button onClick={() => setCurrentView("FindNewGames")}>
          Find New Games
        </button>
        <button onClick={() => setCurrentView("LikedGames")}>
          Liked Games
        </button>
        <button onClick={() => window.open("mailto:jackson.p.rusch@vanderbilt.edu", "_blank")}>
          Feedback
        </button>
        <button onClick={() => {
          window.location.replace(makeBackendURL("logout"))
        }}>
          Logout
      </button>
      </div>
    );
  }

  return (
    <div>
      {currentView === "FindNewGames" ? (
        <GameRating details={{ userID }} />
      ) : (
        <GamesList userID={userID} />
      )}
      <button
        className="changeViewBtn"
        onClick={() =>
          setCurrentView("LandingPage"
          )
        }
      >
        Landing Page
      </button>
    </div>
  );
};

export default App;
