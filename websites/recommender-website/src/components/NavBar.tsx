// Navbar.tsx
import React from "react";
import { makeBackendURL } from "../util";
import './NavBar.css';

interface NavbarProps {
    setCurrentView: React.Dispatch<React.SetStateAction<"LandingPage" | "FindNewGames" | "LikedGames" | "HomePage">>;
}

const Navbar: React.FC<NavbarProps> = ({ setCurrentView }) => {
    return (
        <div className="flex justify-between items-center px-4 py-2 bg-gray-800 text-white">
            <div className="flex-grow">
                <button className="navbar-btn text-s sm:text-sm mr-2 sm:mr-4" onClick={() => setCurrentView("HomePage")}>
                    Home Page
                </button>
                <button className="text-s sm:text-sm mr-2 sm:mr-4" onClick={() => setCurrentView("FindNewGames")}>
                    Find New Games
                </button>
                <button className="text-s sm:text-sm mr-2 sm:mr-4" onClick={() => setCurrentView("LikedGames")}>
                    Liked Games
                </button>
                <button className="text-s sm:text-sm mr-2 sm:mr-4" onClick={() => window.open("mailto:jackson.p.rusch@vanderbilt.edu", "_blank")}>
                    Feedback
                </button>
            </div>
            <div>
                <button className="text-s sm:text-sm justify-right" onClick={() => window.location.replace(makeBackendURL("logout"))}>
                    Logout
                </button>
            </div>
        </div>
    );
};

export default Navbar;