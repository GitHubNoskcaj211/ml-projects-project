import React, { useState } from "react";

function Slider(props) {
  const [value, setValue] = useState(100);

  // Handle slider value change
  const handleChange = (event) => {
    setValue(parseInt(event.target.value, 10));
  };

  const renderLabels = () => {
    if (props.label) {
      // Single label scenario
      return (
        <div style={{ textAlign: "center" }}>
          <span>{props.label}</span>
        </div>
      );
    } else {
      // Two labels scenario
      return (
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <span>{props.leftLabel}</span>
          <span>{props.rightLabel}</span>
        </div>
      );
    }
  };

  return (
    <div style={{ margin: "20px" }}>
      {renderLabels()}
      <input
        type="range"
        min="20"
        max="100"
        step="20"
        value={value}
        onChange={handleChange}
        style={{ width: "100%" }}
      />
      <div style={{ textAlign: "center", marginTop: "10px" }}>{value}%</div>
    </div>
  );
}

export default Slider;
