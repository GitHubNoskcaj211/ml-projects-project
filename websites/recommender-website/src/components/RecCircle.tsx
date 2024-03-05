import React from "react";
import "./RecCircle.css";

interface RecCircleProps {
  value: number;
}

const RecCircle: React.FC<RecCircleProps> = ({ value }) => {
  const radius = 50;
  const circumference = 2 * Math.PI * radius;
  const percentage = value * 100; // Convert to percentage
  const strokeValue = (percentage / 100) * circumference;
  const strokeDasharray = `${strokeValue} ${circumference}`;
  const color = `rgb(${255 * (1 - value)}, ${255 * value}, 0)`;

  return (
    <div className="recCircle">
      <div>
        <p>Reccomendation Score </p>
      </div>
      <svg width="120" height="120" viewBox="0 0 120 120">
        <circle
          className="recCircle-back"
          cx="60"
          cy="60"
          r={radius}
          strokeWidth="10"
        />
        <circle
          className="recCircle-front"
          cx="60"
          cy="60"
          r={radius}
          strokeWidth="10"
          transform="rotate(-90 60 60)"
          style={{
            strokeDasharray,
            stroke: color,
          }}
        />
        <text
          x="50%"
          y="50%"
          dominantBaseline="middle"
          textAnchor="middle"
          className="progressCircle-text"
          style={{
            stroke: color,
          }}
        >
          {`${percentage.toFixed(1)}%`}
        </text>
      </svg>
    </div>
  );
};

export default RecCircle;
