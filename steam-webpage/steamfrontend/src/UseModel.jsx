import React, { useState, useEffect } from "react";
import styles from "./UseModel.module.css";
import SteamButton from "./components/SteamButton";
import Arrow from "./components/Arrow";
import Model from "./components/Model";

const UseModel = () => {
  const [currentStep, setCurrentStep] = useState(1);
  const [selectedButton, setSelectedButton] = useState(3);
  const [sliderValue, setSliderValue] = useState(3);

  useEffect(() => {
    if (currentStep === 2) setSelectedButton(3);
  }, [currentStep]);

  const goToNextStep = () => setCurrentStep((prev) => Math.min(prev + 1, 3));
  const goToPreviousStep = () =>
    setCurrentStep((prev) => Math.max(prev - 1, 1));
  const handleButtonClick = (index) => setSelectedButton(index);
  const handleSliderChange = (event) => setSliderValue(event.target.value);

  const modelTitles = [
    "Initial Model",
    "CNN Model",
    "Stupid Model",
    "Best Model",
  ];
  const modelContents = [
    "It Works",
    "It works well",
    "It does not work well",
    "Our best model so far",
  ];

  return (
    <div className={styles.container}>
      {currentStep === 1 && (
        <div>
          <div className={styles.upperContainer}>
            <h1>Step 1: Log into Steam</h1>
            <p style={{ color: "white" }}>
              We currently only support public accounts.
            </p>
          </div>
          <div className={styles.middleContainer}>
            <SteamButton
              onClick={() => console.log("Sign in through Steam")}
              content="Sign in Through Steam"
              className={`${styles.steamBlueButton} additional-class-for-hover-effect`}
            />
          </div>
          <div className={styles.lowerContainer}>
            <Arrow
              onClick={goToNextStep}
              content="→"
              className={styles.rightButton}
            />
          </div>
        </div>
      )}

      {currentStep === 2 && (
        <div>
          <div className={styles.upperContainer}>
            <h1>Step 2: Choose your Model</h1>
          </div>
          <div className={styles.modelContainer}>
            <div className={styles.leftContainer}>
              {modelTitles.map((title, index) => (
                <Model
                  key={index}
                  onClick={() => handleButtonClick(index)}
                  content={title}
                  className={styles.modelButton}
                />
              ))}
            </div>
            <div className={styles.centerContainer}>
              <p style={{ color: "white" }}>{modelContents[selectedButton]}</p>
            </div>
          </div>
          <div className={styles.lowerContainer}>
            <Arrow
              onClick={goToPreviousStep}
              content="←"
              className={styles.leftButton}
            />
            <Arrow
              onClick={goToNextStep}
              content="→"
              className={styles.rightButton}
            />
          </div>
        </div>
      )}

      {currentStep === 3 && (
        <div>
          <div className={styles.upperContainer}>
            <h1>Step 3: Choose the popularity of your recommendations</h1>
          </div>
          <div className={styles.middleContainer}>
            <span style={{ color: "white" }}>niche</span>
            <input
              type="range"
              min="1"
              max="5"
              value={sliderValue}
              onChange={handleSliderChange}
              className={styles.slider}
            />
            <span style={{ color: "white" }}>popular</span>
            <p style={{ color: "white" }}>
              You have selected: {sliderValue * 20}%
            </p>
          </div>
          <div className={styles.submitContainer}>
            <SteamButton
              onClick={() => console.log("Submit")}
              content="Submit"
              className={styles.submitButton}
            />
          </div>
          <div className={styles.lowerContainer}>
            <Arrow
              onClick={goToPreviousStep}
              content="←"
              className={styles.leftButton}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default UseModel;
