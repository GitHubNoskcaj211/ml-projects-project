// components/SignIn.tsx
import React from "react";
import PublicDirectionsBox from "./publicDirections";
import { makeBackendURL } from "../util";

interface SignInProps {
  showPublicProfileWarning: boolean;
  setShowPublicProfileWarning: React.Dispatch<React.SetStateAction<boolean>>;
}

const SignIn: React.FC<SignInProps> = ({ showPublicProfileWarning, setShowPublicProfileWarning }) => {
  return (
    <div className="container signInContainer">
      {
        /* Popup Directions Box*/
        showPublicProfileWarning && (
          <PublicDirectionsBox isOpen={showPublicProfileWarning} onClose={() => setShowPublicProfileWarning(false)} />
        )
      }
      <button onClick={() => (location.href = makeBackendURL("/login"))}>
        Sign in through Steam
      </button>
    </div>
  );
};

export default SignIn;
