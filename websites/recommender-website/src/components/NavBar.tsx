// Navbar.tsx
import React from "react";
import { makeBackendURL } from "../util";
import './NavBar.css';

interface NavbarProps {
    setCurrentView: React.Dispatch<React.SetStateAction<"LandingPage" | "FindNewGames" | "Interactions" | "HomePage">>;
}

const Navbar: React.FC<NavbarProps> = ({ setCurrentView }) => {
    return (
        <div className="flex justify-between items-center px-4 py-2 bg-gray-800 text-white">
            <div className="flex-grow">
                <button className="text-sm navbar-btn md:text-lg mr-2 sm:mr-4" onClick={() => setCurrentView("HomePage")}>
                    Home Page
                </button>
                <button className="text-sm md:text-lg mr-2 sm:mr-4" onClick={() => setCurrentView("FindNewGames")}>
                    Find New Games
                </button>
                <button className="text-sm md:text-lg mr-2 sm:mr-4" onClick={() => setCurrentView("Interactions")}>
                    Interactions
                </button>
                <button className="text-sm md:text-lg mr-2 sm:mr-4" onClick={() => window.open("mailto:jackson.p.rusch@vanderbilt.edu", "_blank")}>
                    Feedback
                </button>
            </div>
            <div>
                <button className="text-sm md:text-lg justify-right" onClick={() => window.location.replace(makeBackendURL("logout"))}>
                    Logout
                </button>
            </div>
        </div>
    );
};

export default Navbar;
