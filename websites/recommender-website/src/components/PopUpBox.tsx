import React from 'react';
import './PopUpBox.css';

interface PopUpBox {
    isOpen: boolean;
    onClose: () => void;
    message: string;
}

const PopUpBox: React.FC<PopUpBox> = ({ isOpen, onClose }) => {
    if (!isOpen) return null;

    return (
        <div className="full-page-popup">
          <div className="popup-inner">
            <button className="popup-close-btn" onClick={onClose}>X</button>
            <h1>Steam Recommendations</h1>
            <p>Get personalized game recommendations! Swipe right if you like a game, or swipe left if it's not your cup of tea. Use the arrow icons below each game for your selection.</p>
            
            <p>Start by exploring the game below and select your preferences to get recommendations tailored just for you.</p>

            <div>
               <span>Swipe Left if you don't like it! </span><span>👈</span>
            </div>
            <div>
              <span>Swipe Right if you like it! </span> <span>👉</span>
            </div>  
          </div>
        </div>
      );
    };      


export default PopUpBox;
