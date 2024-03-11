import React from 'react';
import './PopUpBox.css';

interface PublicDirectionsBox {
    isOpen: boolean;
    onClose: () => void;
}

const PopUpDirections: React.FC<PublicDirectionsBox> = ({ isOpen, onClose }) => {
    if (!isOpen) return null;

    return (
        <div className="full-page-popup">
            <div className="popup-inner">
                <button className="popup-close-btn" onClick={onClose}>X</button>
                <h1>Public Profile?</h1>
                <p>It looks like we're having some trouble viewing your profile...</p>

                <p>Let's check if you're profile is public! If it's not, we can't retrieve your data</p>
                <div>
                    <span>Please go to the official Steam website and change the setting <a href="https://help.steampowered.com/en/faqs/view/588C-C67D-0251-C276">here</a>.</span>
                </div>
            </div>
        </div>
    );
};


export default PopUpDirections;
