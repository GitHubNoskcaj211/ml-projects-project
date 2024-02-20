// Home.jsx
import React from "react";
import StandardButton from "./components/StandardButton.jsx";
import styles from "./Home.module.css";

function Home() {
  return (
    <div className={styles.container} style={{ textAlign: "center" }}>
      <h1 className={styles.header}>Steam Game Recommendation System</h1>
      <p className={styles.paragraph}>
        We weren't satisfied with Steam's native game recommendations so we
        decided to build our own.
      </p>
      <StandardButton
        onClick={() => console.log("Hey")}
        content="Get Started"
        className="btn btn-outline-primary"
      />
    </div>
  );
}

export default Home;
