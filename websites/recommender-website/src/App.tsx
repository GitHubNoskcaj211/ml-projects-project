import React, { useEffect, useState } from "react";
import GameRating from "./components/GameRating";
import GamesList from "./components/GamesList";
import Navbar from "./components/NavBar";
import SignIn from "./components/SignIn";
import Loading from "./components/Loading";
import HomePage from "./components/HomePage";
import "./App.css";

import { makeBackendURL } from "./util";

const App: React.FC = () => {
  const [userID, setUserID] = useState<string | undefined | null>(undefined);
  const [showPublicProfileWarning, setShowPublicProfileWarning] = useState(false);
  const [currentView, setCurrentView] = useState<
    "LandingPage" | "FindNewGames" | "LikedGames" | "HomePage"
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
      console.log(data)
      setUserID(data.id);
      setCurrentView("HomePage");
    })();
    return () => {
      controller.abort();
    };
  }, []);

  const closePopup = () => {
    setShowPublicProfileWarning(false);
  };

  if (userID === undefined) {
    return <Loading />
  }

  if (userID === null) {
    return (
      <SignIn
        showPublicProfileWarning={showPublicProfileWarning}
        setShowPublicProfileWarning={setShowPublicProfileWarning}
      />
    );
  }


  return (
    <div>
      <Navbar setCurrentView={setCurrentView}/>
      {currentView === "HomePage" && (
        <HomePage  />
      )}
      {currentView === "FindNewGames" && (
        <GameRating details={{ userID }} />
      )}
      {currentView === "LikedGames" && (
        <GamesList userID={userID} />
      )}
      
    </div>
  );
};

export default App;
