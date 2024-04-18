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
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex justify-center items-center px-4">
            <div className="popup-inner bg-white p-4 rounded-md shadow-lg max-w-full sm:max-w-lg mx-auto">
                <button className="popup-close-btn absolute top-2 right-2" onClick={onClose}>X</button>
                <h1 className="text-xl sm:text-2xl md:text-3xl lg:text-4xl font-bold mb-3 sm:mb-2">Steam Recommendations</h1>
                <p className="text-lg sm:text-sm mb-3 sm:mb-2">Get personalized game recommendations! {isMobile ? "Swipe right" : "Click the right arrow"} if you like a game, or {isMobile ? "swipe left" : "click the left arrow"} if it's not your cup of tea. </p>

                <p className="text-lg sm:text-sm mb-3 sm:mb-2">Start by exploring the game below and select your preferences to get recommendations tailored just for you.</p>

                {isMobile ? (
                    <>
                        <div className="text-xl sm:text-sm mb-2">
                            <span>Swipe Left if you don't like it! </span><span>👈</span>
                        </div>
                        <div className="text-xl sm:text-sm mb-2">
                            <span>Swipe Right if you like it! </span> <span>👉</span>
                        </div>
                    </>
                ) : (
                    <>
                        <div className="text-xl sm:text-sm mb-2">
                            <span>Hit the left arrow for dislikes. </span><span>⬅️</span>
                        </div>
                        <div className="text-xl sm:text-sm mb-2">
                            <span>Hit the right arrow for likes.</span> <span>➡️</span>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
};

export default PopUpBox;
