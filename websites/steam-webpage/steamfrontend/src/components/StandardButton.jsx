import React from "react";

const StandardButton = ({
  content,
  onClick,
  className = "btn btn-outline-light",
}) => {
  return (
    <button type="button" className={className} onClick={onClick}>
      {content}
    </button>
  );
};

export default StandardButton;
