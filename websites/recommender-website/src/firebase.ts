import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { collection, getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyDoV6rP_Vo0t9ynB2wUs4wCotfMzpWUOdk",
  authDomain: "steam-game-recommender-415605.firebaseapp.com",
  projectId: "steam-game-recommender-415605",
  storageBucket: "steam-game-recommender-415605.appspot.com",
  messagingSenderId: "1018682103642",
  appId: "1:1018682103642:web:e8acf59ad92bae88f54563",
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

export const auth = getAuth(app);
export const db = getFirestore(app);

export const getInteractionsCollection = () =>
  collection(db, "interactions", "data", auth.currentUser!.uid);
