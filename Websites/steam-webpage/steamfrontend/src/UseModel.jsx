import React, { useState } from "react";
import styles from "./UseModel.module.css";
import Button from "./components/StandardButton";
import Slider from "./components/Slider";

const UseModel = () => {
  const [selectedButton, setSelectedButton] = useState(3);

  const handleButtonClick = (index) => setSelectedButton(index);

  const modelTitles = [
    "Initial Model",
    "CNN Model",
    "Stupid Model",
    "Best Model",
  ];

  return (
    <div className={styles.container}>
      <div className={styles.upperContainer}>
        <h1>Log into Steam</h1>
        <p style={{ color: "white" }}>
          We currently only support public accounts.
        </p>
      </div>
      <div className={styles.mainContainer}>
        <div className={styles.modelContainer}>
          {modelTitles.map((title, index) => (
            <Button
              key={index}
              onClick={() => handleButtonClick(index)}
              content={title}
            />
          ))}
        </div>
        <div className={styles.rightContainer}>
          <div className={styles.sliderContainer}>
            <Slider leftLabel="Niche" rightLabel="Popular" />
            <Slider label="Model Depth" />
          </div>
          <div className={styles.gamesContainer}>
            <p>"placeholder"</p>
          </div>
        </div>
      </div>
      <div className={styles.lowerContainer}>
        <Button
          onClick={() => {}}
          content="→"
          className="btn btn-outline-warning"
        />
      </div>
    </div>
  );
};

export default UseModel;
