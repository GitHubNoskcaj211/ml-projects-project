import React, { useEffect, useState } from "react";
import GameRating from "./GameRating";
import PublicDirectionsBox from "./components/publicDirections";

import { makeBackendURL } from "./util";

const App: React.FC = () => {
  const [userID, setUserID] = useState<string | undefined | null>(undefined);
  const [showPopup, setShowPopup] = useState(false);

  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      
      if (showPopup && event.key === "Escape") {
        closePopup();
      }
    };

    window.addEventListener("keydown", handleKeyPress);
    return () => {
      window.removeEventListener("keydown", handleKeyPress);
    };
  }, [showPopup]);


  useEffect(() => {
    (async () => {
      const res = await fetch(makeBackendURL("init_user"), {
        mode: "cors",
        credentials: "include",
      });
      if (res.status === 401) {
        setUserID(null);
        const attempts = parseInt(localStorage.getItem('loginAttempts') || '0') + 1;
        localStorage.setItem('loginAttempts', attempts.toString());
        if (attempts > 2) {
          setShowPopup(true);
        }
        return;
      }
      const data = await res.json();
      setUserID(data.id);
    })();
  }, []);

  useEffect(() => {
    const attempts = parseInt(localStorage.getItem('loginAttempts') || '0');
    if (attempts > 2) {
      setShowPopup(true);
    }
  }, []);

  const closePopup = () => {
    setShowPopup(false);
  };

  if (userID === undefined) {
    return <div className="container">Loading...</div>;
  }

  if (userID === null) {
    return (
      <div className="container signInContainer">
        {
          /* Popup Directions Box*/
          showPopup && (
            <PublicDirectionsBox
              isOpen={showPopup}
              onClose={closePopup}
            />
          )
        }
        <button onClick={() => (location.href = makeBackendURL("/login"))} >
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
