import React, { useEffect, useState } from "react";
import { onAuthStateChanged, signInWithCustomToken } from "firebase/auth";
import { auth } from "./firebase";
import GameRating from "./components/GameRating";
import GamesList from "./components/GamesList";
import { Navbar, NavbarView } from "./components/NavBar";
import SignIn from "./components/SignIn";
import Loading from "./components/Loading";
import HomePage from "./components/HomePage";
import "./App.css";

import { backendAuthFetch } from "./util";

const App: React.FC = () => {
  const [userID, setUserID] = useState<string | undefined | null>(undefined);
  const [userInited, setUserInited] = useState(false);
  const [showPublicProfileWarning, setShowPublicProfileWarning] =
    useState(false);
  const [currentView, setCurrentView] = useState<NavbarView>(
    NavbarView.LandingPage
  );

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
    const cleanup = onAuthStateChanged(auth, (user) => {
      setUserID(user?.uid ?? null);
    });

    (async () => {
      const urlParams = new URLSearchParams(window.location.search);
      const token = urlParams.get("token");
      if (token === null) return;

      urlParams.delete("token");
      await signInWithCustomToken(auth, token);
    })();
    return cleanup;
  }, []);

  useEffect(() => {
    if (userID === null) return;
    const controller = new AbortController();
    (async () => {
      const res = await backendAuthFetch("init_user", {
        method: "POST",
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
      } else if (res.status !== 200) {
        setUserID(null);
        return;
      }
      setUserInited(true);
      setCurrentView(NavbarView.HomePage);
    })();

    return () => {
      controller.abort();
    };
  }, [userID]);

  const closePopup = () => {
    setShowPublicProfileWarning(false);
  };

  if (userID === undefined || !userInited) {
    return <Loading />;
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
      <Navbar setCurrentView={setCurrentView} />
      {currentView === NavbarView.HomePage && <HomePage />}
      {currentView === NavbarView.FindNewGames && (
        <GameRating details={{ userID }} />
      )}
      {currentView === NavbarView.Interactions && <GamesList userID={userID} />}
    </div>
  );
};

export default App;
