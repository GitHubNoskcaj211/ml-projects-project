import React from "react";

const Button = ({ content, onClick }) => {
  return (
    <button
      type="button"
      className="btn btn-outline-light btn-lg btn-block"
      onClick={onClick}
    >
      {content}
    </button>
  );
};

export default Button;
