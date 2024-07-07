import React, { useState, useEffect } from "react";
import "./GameRating.css";
import RecCircle from "./RecCircle";
import PopUpBox from "./PopUpBox";
import {
  fetchGameRecommendations,
  RecResponse,
  RecommendationsResponse,
} from "./GetRecs";
import { backendAuthFetch, delay } from "../util";

interface Game {
  userID: string;
  gameID?: string;
  userSelection?: boolean;
  timeSpent?: number;
}

interface GameRatingProps {
  details: Game;
}

const REQ_BATCH_SIZE = 10;
const BUFFER_SIZE = 20;
const MIN_HORIZONTAL_SWIPE_COLOR_CHANGE = window.innerWidth / 4;
const HORIZONTAL_SWIPE_THRESHOLD = window.innerWidth / 2;
const VERTICAL_SWIPE_THRESHOLD = 50;
const MAX_SWIPE_COLOR_ALPHA = 0.25;
const SWIPE_COLOR_CHANGE_TIME_MS = 500;

const GameRating: React.FC<GameRatingProps> = ({ details }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [showPopup, setShowPopup] = useState(true);
  const [startTime, setStartTime] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [interactionAttempts, setInteractionAttempts] = useState(0);
  const [recommendations, setRecommendations] = useState<RecResponse[]>([]);
  const [steamLinkClicked, setSteamLinkClicked] = useState(false);
  const [swipeDirection, setSwipeDirection] = useState<"left" | "right" | null>(null);
  const [swipeProgress, setSwipeProgress] = useState(0); // Percentage of swipe progress

  const handleSteamLinkClicked = () => {
    setSteamLinkClicked(true);
  };

  const runGamesProcess = async () => {
    console.log("Effect for fetchGameRecommendations running", details.userID);
    while (true) {
      try {
        const exclude_game_ids = recommendations.map((rec) => rec.game_id);
        let resp: RecommendationsResponse;
        do {
          resp = await fetchGameRecommendations(REQ_BATCH_SIZE, exclude_game_ids);
          if (resp.recommendations.length === 0) {
            await delay(1000);
          }
        } while (resp.recommendations.length == 0);
        setRecommendations((prev) => prev.concat(resp.recommendations));
        setLoading(false);
        return
      } catch (e) {
        console.error("Error fetching games or game info. Trying again:", e);
      }
    }
  };

  useEffect(() => {
    if (showPopup) {
      const handleKeyPress = (event: KeyboardEvent) => {
        if (event.key === "Escape") {
          closePopup();
        }
      };
      window.addEventListener("keydown", handleKeyPress);
      return () => {
        window.removeEventListener("keydown", handleKeyPress);
      };
    }

    if (loading || interactionAttempts > 1) {
      return;
    }

    const handleKeyPress = async (event: KeyboardEvent) => {
      if (loading) return;

      if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") {
        return;
      }
      await handleUserLikeDislike(event.key === "ArrowRight");
    };

    // Function to handle touch events for swipe
    const handleSwipe = async (startX: number, endX: number, startY: number, endY: number) => {
      if (Math.abs(startY - endY) >= VERTICAL_SWIPE_THRESHOLD) {
        return;
      }
      if (endX - startX > HORIZONTAL_SWIPE_THRESHOLD) {
        await handleUserLikeDislike(true);
      } else if (startX - endX > HORIZONTAL_SWIPE_THRESHOLD) {
        await handleUserLikeDislike(false);
      }
    };
    let touchStartX = 0;
    let touchStartY = 0;
    let isTouchInProgress = false;

    const onTouchStart = (event: TouchEvent) => {
      if (!isTouchInProgress) {
        touchStartX = event.touches[0].clientX;
        touchStartY = event.touches[0].clientY;
        isTouchInProgress = true;
      }
    };

    const onTouchMove = (event: TouchEvent) => {
      if (isTouchInProgress) {
        const touchMoveX = event.touches[0].clientX;
        const touchMoveY = event.touches[0].clientY;
        if (Math.abs(touchStartY - touchMoveY) >= VERTICAL_SWIPE_THRESHOLD) {
          setSwipeDirection(null);
          setSwipeProgress(0);
          return;
        }
        setSwipeDirection(touchMoveX - touchStartX > MIN_HORIZONTAL_SWIPE_COLOR_CHANGE ? "right" : touchStartX - touchMoveX > MIN_HORIZONTAL_SWIPE_COLOR_CHANGE ? "left" : null);
        setSwipeProgress(Math.min(Math.max(Math.abs(touchMoveX - touchStartX) - MIN_HORIZONTAL_SWIPE_COLOR_CHANGE, 0.0) / (HORIZONTAL_SWIPE_THRESHOLD - MIN_HORIZONTAL_SWIPE_COLOR_CHANGE), 1.0) * MAX_SWIPE_COLOR_ALPHA);
      }
    };

    const onTouchEnd = async (event: TouchEvent) => {
      if (isTouchInProgress) {
        const touchEndX = event.changedTouches[0].clientX;
        const touchEndY = event.changedTouches[0].clientY;
        await handleSwipe(touchStartX, touchEndX, touchStartY, touchEndY);
        isTouchInProgress = false;
        setSwipeDirection(null);
        setSwipeProgress(0);
      }
    };

    window.addEventListener("keydown", handleKeyPress);
    window.addEventListener("touchstart", onTouchStart);
    window.addEventListener("touchmove", onTouchMove);
    window.addEventListener("touchend", onTouchEnd);
    return () => {
      window.removeEventListener("keydown", handleKeyPress);
      window.removeEventListener("touchstart", onTouchStart);
      window.removeEventListener("touchmove", onTouchMove);
      window.removeEventListener("touchend", onTouchEnd);
    };
  }, [
    showPopup,
    loading,
    startTime,
    currentIndex,
    steamLinkClicked,
    interactionAttempts,
    recommendations.length,
  ]);

  const handleUserLikeDislike = async (userLiked: boolean) => {
    setInteractionAttempts((prev) => prev + 1);
    if (interactionAttempts > 1) {
      return;
    }
    if (startTime === null) {
      throw Error("startTime is null");
    }
    const timeSpent = (Date.now() - startTime) / 1000;

    console.assert(currentIndex < recommendations.length);
    const rec = recommendations[currentIndex];
    const newIndex = currentIndex + 1;
    setSwipeDirection(userLiked ? 'right' : 'left');
    setSwipeProgress(MAX_SWIPE_COLOR_ALPHA);
    await backendAuthFetch("add_interaction", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        rec_model_name: rec.model_name,
        rec_model_save_path: rec.model_save_path,
        num_game_interactions_local: rec.num_game_interactions_local.toString(),
        num_game_owned_local: rec.num_game_owned_local.toString(),
        num_game_interactions_external:
          rec.num_game_interactions_external.toString(),
        num_game_owned_external: rec.num_game_owned_external.toString(),
        game_id: rec.game_id.toString(),
        user_liked: userLiked.toString(),
        time_spent: timeSpent,
        steam_link_clicked: steamLinkClicked.toString(),
      }),
    });
    // TODO have this run on updates to the # interactions documents.
    backendAuthFetch("check_refresh_all_recommendation_queues", {
      method: "POST",
    });
    if (newIndex >= recommendations.length) {
      setLoading(true);
    }
    setInteractionAttempts(0);
    setCurrentIndex(newIndex);
    setStartTime(Date.now());
    setSteamLinkClicked(false);
    setTimeout(() => {
      setSwipeDirection(null);
      setSwipeProgress(0);
    }, SWIPE_COLOR_CHANGE_TIME_MS);
  }

  useEffect(() => {
    const swipeProgressString = swipeProgress.toString();
    document.documentElement.style.setProperty('--swipe-progress', swipeProgressString);
  }, [swipeProgress]);

  useEffect(() => {
    if (recommendations.length - currentIndex >= BUFFER_SIZE) {
      return;
    }
    runGamesProcess();
  }, [currentIndex]);

  useEffect(() => {
    if (recommendations.length - currentIndex === 0) {
      setStartTime(null);
    } else {
      setStartTime((prev) => (prev ? prev : Date.now()));
    }
  }, [currentIndex, recommendations.length]);

  const handleUndo = () => {
    console.assert(currentIndex > 0);
    setCurrentIndex((prev) => prev - 1);
    setStartTime(Date.now());
  };

  const closePopup = () => {
    setShowPopup(false);
    setStartTime(Date.now());
  };

  if (loading || interactionAttempts > 1) {
    backendAuthFetch("check_refresh_all_recommendation_queues", { // TODO have this run on an init-ed user (update to users_games / friends).
      method: "POST",
    });
    return <div>Loading...</div>;
  }

  return (
    <div className={`container ${swipeDirection === "right" ? "swipeRight" : swipeDirection === "left" ? "swipeLeft" : ""}`}>
      {
        /* Popup Directions Box*/
        showPopup && currentIndex === 0 && (
          <PopUpBox isOpen={showPopup} onClose={closePopup} />
        )
      }

      <div className="contentContainer">
        {/* Game Title */}
        <div className="title box">
          <a
            href={`https://store.steampowered.com/app/${recommendations[currentIndex].game_id}`}
            target="_blank"
            rel="noopener noreferrer"
            onClick={handleSteamLinkClicked}
          >
            <h1>{recommendations[currentIndex].name}</h1>
          </a>
        </div>
        {/* Price */}
        <div className="price box">
          <h2 className='font-bold text-2xl'>Price: ${recommendations[currentIndex].price}</h2>
        </div>
        <div className="secondRow">
          {/* Image */}
          <div className="image box">
            <img
              src={`https://cdn.akamai.steamstatic.com/steam/apps/${recommendations[currentIndex].game_id}/header.jpg`}
              alt={recommendations[currentIndex].name}
            />
          </div>

          {/* RecCircle */}
          <div className="rec box">
            <RecCircle
              value={recommendations[currentIndex].avgReviewScore || 0}
              num_reviewers={
                recommendations[currentIndex].numReviews || 0
              }
            />
          </div>
        </div>

        {/* Game Description */}
        <div className="game box">
          <p>{recommendations[currentIndex].description}</p>
        </div>

        {/* Genres */}
        <div className="genre box">
          <h2>Genres</h2>
        </div>
        <div className="flex flex-wrap gap-2 justify-center text-gray-400">
          {recommendations[currentIndex].genres.map(
            (genre: string, index: number) => (
              <button key={index} disabled>
                {genre}
              </button>
            )
          )}
        </div>

        {/* Tags */}
        <div className="tag box">
          <h2>Tags</h2>
        </div>
        <div className="flex flex-wrap gap-2 justify-center text-gray-500 ">
          {recommendations[currentIndex].tags.map(
            (tag: string, index: number) => (
              <button key={index} disabled>
                {tag}
              </button>
            )
          )}
        </div>

        {/* Undo Button */}
        <div className="undoContainer">
          {currentIndex > 0 && (
            <button className="undoButton" onClick={handleUndo}>
              Go Back
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default GameRating;
