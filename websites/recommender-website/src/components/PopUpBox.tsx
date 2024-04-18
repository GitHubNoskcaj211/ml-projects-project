import React from 'react';
import './PopUpBox.css';

interface PopUpBoxProps {
    isOpen: boolean;
    onClose: () => void;
}

const PopUpBox: React.FC<PopUpBoxProps> = ({ isOpen, onClose }) => {
    if (!isOpen) return null;

    // Simple mobile detection
    const isMobile = /Mobi|Android/i.test(navigator.userAgent);

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex justify-center items-center">
            <div className="popup-inner bg-white p-4 rounded-md shadow-lg max-w-md mx-auto">
                <button className="popup-close-btn absolute top-2 right-2" onClick={onClose}>X</button>
                <h1 className="text-4xl sm:text-xl font-bold">Steam Recommendations</h1>
                <p>Get personalized game recommendations! {isMobile ? "Swipe right" : "Click the right arrow"} if you like a game, or {isMobile ? "swipe left" : "click the left arrow"} if it's not your cup of tea. </p>

                <p>Start by exploring the game below and select your preferences to get recommendations tailored just for you.</p>

                {isMobile ? (
                    <>
                        <div>
                            <span>Swipe Left if you don't like it! </span><span>👈</span>
                        </div>
                        <div>
                            <span>Swipe Right if you like it! </span> <span>z👉</span>
                        </div>
                    </>
                ) : (
                    <>
                        <div>
                            <span>Hit the left arrow for dislikes. </span><span>⬅️</span>
                        </div>
                        <div>
                            <span>Hit the right arrow for likes.</span> <span>➡️</span>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
};

export default PopUpBox;
