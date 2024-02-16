import React from "react";
import { BrowserRouter as Router, Route, Link } from "react-router-dom";
import "./App.css";
import Home from "./Home";
import UseModel from "./UseModel";

function App() {
  return (
    <>
      <div>
        <Home />
      </div>
      <div>
        <UseModel />
      </div>
    </>
  );
}

export default App;
