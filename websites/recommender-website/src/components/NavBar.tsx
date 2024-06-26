// Navbar.tsx
import React from "react";
import "./NavBar.css";
import { signOut } from "firebase/auth";
import { auth } from "../firebase";

export enum NavbarView {
  LandingPage,
  FindNewGames,
  Interactions,
  HomePage,
}

interface NavbarProps {
  setCurrentView: React.Dispatch<React.SetStateAction<NavbarView>>;
}

export const Navbar: React.FC<NavbarProps> = ({ setCurrentView }) => {
  return (
    <div className="flex justify-between items-center px-4 py-2 bg-gray-800 text-white">
      <div className="flex-grow">
        <button
          className="text-sm navbar-btn md:text-lg mr-2 sm:mr-4"
          onClick={() => setCurrentView(NavbarView.HomePage)}
        >
          Home Page
        </button>
        <button
          className="text-sm md:text-lg mr-2 sm:mr-4"
          onClick={() => setCurrentView(NavbarView.FindNewGames)}
        >
          Find New Games
        </button>
        <button
          className="text-sm md:text-lg mr-2 sm:mr-4"
          onClick={() => setCurrentView(NavbarView.Interactions)}
        >
          Interactions
        </button>
        <button
          className="text-sm md:text-lg mr-2 sm:mr-4"
          onClick={() =>
            window.open("mailto:jackson.p.rusch@vanderbilt.edu", "_blank")
          }
        >
          Feedback
        </button>
      </div>
      <div>
        <button
          className="text-sm md:text-lg justify-right"
          onClick={() => signOut(auth)}
        >
          Logout
        </button>
      </div>
    </div>
  );
};
