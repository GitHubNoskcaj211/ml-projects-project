import React, { useState, useEffect } from "react";
import styles from "./UseModel.module.css";
import Button from "./components/StandardButton";
import Slider from "./components/Slider";

const UseModel = () => {
  const [currentStep, setCurrentStep] = useState(1);
  const [selectedButton, setSelectedButton] = useState(3);

  useEffect(() => {
    if (currentStep === 2) setSelectedButton(3);
  }, [currentStep]);

  const goToNextStep = () => setCurrentStep((prev) => Math.min(prev + 1, 3));
  const goToPreviousStep = () =>
    setCurrentStep((prev) => Math.max(prev - 1, 1));
  const handleButtonClick = (index) => setSelectedButton(index);

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
            <h1>Log into Steam</h1>
            <p style={{ color: "white" }}>
              We currently only support public accounts.
            </p>
          </div>
          <div className={styles.middleContainer}>
            <Button
              onClick={() => console.log("Sign in through Steam")}
              content="Sign in Through Steam"
              className="btn btn-outline-info"
            />
          </div>
          <div className={styles.lowerContainer}>
            <Button
              onClick={goToNextStep}
              content="→"
              className="btn btn-outline-warning"
            />
          </div>
        </div>
      )}

      {currentStep === 2 && (
        <div>
          <div className={styles.upperContainer}>
            <h1>Customise your Search</h1>
          </div>
          <div className={styles.modelContainer}>
            <div className={styles.leftContainer}>
              <div class="d-grid gap-2">
                {modelTitles.map((title, index) => (
                  <Button
                    onClick={() => handleButtonClick(index)}
                    content={title}
                  />
                ))}
              </div>
            </div>
            <div className={styles.centerContainer}>
              <Slider leftLabel="Niche" rightLabel="Popular" />
              <Slider label="Model Depth" />
            </div>
          </div>
          <div className={styles.lowerContainer}>
            <Button
              onClick={goToPreviousStep}
              content="←"
              className="btn btn-outline-warning"
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default UseModel;
