// Home.jsx
import React from "react";
import Button from "./components/GetStartedButton.jsx";
import styles from "./Home.module.css";

function Home() {
  return (
    <div className={styles.container} style={{ textAlign: "center" }}>
      <h1 className={styles.header}>Steam Game Recommendation System</h1>
      <p className={styles.paragraph}>
        We weren't satisfied with Steam's native game recommendations so we
        decided to build our own.
      </p>
      <Button
        onClick={() => console.log("Hey")}
        content="Get Started"
        className={`btn ${styles.customButton}`}
      />
    </div>
  );
}

export default Home;
