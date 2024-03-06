import React, { useState, useEffect } from "react";
import "./GameRating.css"; // Import the CSS file
import RecCircle from "./components/RecCircle";
import PopupBox from "./components/PopUpBox";

interface Game {
  name: string;
  rating: number;
  userSelection?: number;
  timeSpent?: number;
}

interface GameRatingProps {
  games: Game[];
}

const GameRating: React.FC<GameRatingProps> = ({ games }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [history, setHistory] = useState<number[]>([]);
  const [finalGames, setFinalGames] = useState<Game[]>([]);
  const [showPopup, setShowPopup] = useState(true);
  const [startTime, setStartTime] = useState<number | null>(null);


  const [descriptions] = useState<string[]>([
    "Baldur’s Gate 3 is a story-rich, party-based RPG set in the universe of Dungeons & Dragons, where your choices shape a tale of fellowship and betrayal, survival and sacrifice, and the lure of absolute power.",
    "For over two decades, Counter-Strike has offered an elite competitive experience, one shaped by millions of players from across the globe. And now the next chapter in the CS story is about to begin. This is Counter-Strike 2.",
    "Cut-throat multiplayer running game that pits 4 players against each other, locally and/or online. Run, jump, swing around, and use devious weapons and pick-ups to knock opponents off-screen! One of the most competitive games you'll ever play.",
  ]);

  const genres: string[][] = [
    ["Adventure", "RPG", "Strategy"],
    ["Action", "Free to Play"],
    ["Action", "Casual", "Indie", "Racing", "Sports"],
  ];

  const gameID: string[] = ["1086940", "730", "207140"];

  const tags: string[][] = [
    [
      "RPG",
      "Choices Matter",
      "Story Rich",
      "Character Customization",
      "Turn-Based Combat",
      "Dungeons & Dragons",
      "Adventure",
      "CRPG",
      "Fantasy",
      "Online Co-Op",
      "Multiplayer",
      "Romance",
      "Strategy",
      "Singleplayer",
      "Co-op Campaign",
      "Class-Based",
      "Sexual Content",
      "Dark Fantasy",
      "Combat",
      "Controller",
    ],
    [
      "FPS",
      "Shooter",
      "Multiplayer",
      "Competitive",
      "Action",
      "Team-Based",
      "eSports",
      "Tactical",
      "First-Person",
      "PvP",
      "Online Co-Op",
      "Co-op",
      "Strategy",
      "Military",
      "War",
      "Difficult",
      "Trading",
      "Realistic",
      "Fast-Paced",
      "Moddable",
    ],
    [
      "Multiplayer",
      "Racing",
      "Local Multiplayer",
      "Indie",
      "Competitive",
      "Fast-Paced",
      "Platformer",
      "Action",
      "2D",
      "4 Player Local",
      "Funny",
      "Parkour",
      "Sports",
      "Controller",
      "Local Co-Op",
      "Co-op",
      "Level Editor",
      "Singleplayer",
      "Arcade",
      "Superhero",
    ],
  ];

  useEffect(() => {
    setFinalGames(games.map((game) => ({ ...game })));
  }, [games]);

  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (startTime && (event.key === "ArrowRight" || event.key === "ArrowLeft")) {
        const selection = event.key === "ArrowRight" ? 1 : 0;
        const updatedGames = [...finalGames];
        const timeSpentCurrent = (Date.now() - startTime) / 1000;

        if (currentIndex < games.length) {
          updatedGames[currentIndex] = {
            ...updatedGames[currentIndex],
            userSelection: selection,
            timeSpent: timeSpentCurrent
          };
          setFinalGames(updatedGames);
        }

        setHistory((prev) => [...prev, currentIndex]);
        setCurrentIndex(currentIndex + 1);
        setStartTime(Date.now());
      }
    };

    window.addEventListener("keydown", handleKeyPress);

    return () => {
      window.removeEventListener("keydown", handleKeyPress);
    };
  }, [currentIndex, finalGames, games.length, startTime]);

  const handleUndo = () => {
    if (history.length > 0) {
      const previousIndex = history.pop()!;
      setHistory([...history]);
      setCurrentIndex(previousIndex);
    }
  };

  const closePopup = () => {
    setShowPopup(false);
    setStartTime(Date.now());
  };


  return (
    <div className="container">
      {/* Popup Directions Box*/
        showPopup && currentIndex === 0 && (
          <PopupBox
            isOpen={showPopup}
            message="Welcome to Game Rating! Close this to start."
            onClose={closePopup}
          />
        )}

      {currentIndex < games.length ? (
        <div className="contentContainer">
          {/* Game Title */}
          <div className="title box">
            <h1>{finalGames[currentIndex]?.name}</h1>
          </div>

          {/* Image */}
          <div className="image box">
            <img
              src={`https://cdn.akamai.steamstatic.com/steam/apps/${gameID[currentIndex]}/header.jpg`}
              alt={finalGames[currentIndex]?.name}
            />
          </div>

          {/* RecCircle */}
          <div className="rec box">
            <RecCircle value={finalGames[currentIndex]?.rating || 0} />
          </div>

          {/* Game Description */}
          <div className="game box">
            <p>{descriptions[currentIndex]}</p>
          </div>

          {/* Genres */}
          <div className="genre box">
            <h2>Genres</h2>
            <div className="genreButtons">
              {genres[currentIndex].map((genre, index) => (
                <button key={index} disabled>
                  {genre}
                </button>
              ))}
            </div>
          </div>

          {/* Tags */}
          <div className="tag box">
            <h2>Tags</h2>
            <div className="tagButtons">
              {tags[currentIndex].map((tag, index) => (
                <button key={index} disabled>
                  {tag}
                </button>
              ))}
            </div>
          </div>

          {/* Undo Button */}
          <div className="undoContainer">
            {history.length > 0 && (
              <button className="undoButton" onClick={handleUndo}>
                Go Back
              </button>
            )}
          </div>
        </div>
      ) : (
        <div className="finalPage">
          <h1>Thank you!</h1>
          <pre>{JSON.stringify(finalGames, null, 2)}</pre>
        </div>
      )}
    </div>
  );

};

export default GameRating;
