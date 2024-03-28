import React, { useEffect, useState } from "react";
import { fetchGameInfo, GameInfo } from "./components/GetGameDetails";
import { makeBackendURL } from "./util";
import "./GamesList.css";

interface GamesListProps {
  userID: string;
}

interface Interaction {
  game_id: number;
  user_liked: boolean;
}

const GamesList: React.FC<GamesListProps> = ({ userID }) => {
  const [gamesLikedInfo, setGamesLikedInfo] = useState<GameInfo[] | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    (async () => {
      const resp = await fetch(
        makeBackendURL("get_all_interactions_for_user"),
        {
          credentials: "include",
          signal: controller.signal,
        }
      );
      const data = await resp.json();
      const interactions = data.interactions;
      const gamesLiked: number[] = interactions
        .filter((interaction: Interaction) => interaction.user_liked)
        .map((interaction: Interaction) => interaction.game_id);
      const promises = gamesLiked.map((id) => fetchGameInfo(id));
      const gamesInfo = await Promise.all(promises);

      setGamesLikedInfo(gamesInfo);
    })();
    return () => {
      controller.abort();
    };
  }, [userID]);

  if (gamesLikedInfo === null) {
    return "Loading...";
  }

  return (
    <div>
      <table id="gamesTable">
        <thead>
          <tr>
            <th>Game Name</th>
            <th>Price</th>
          </tr>
        </thead>
        <tbody>
          {gamesLikedInfo.map((gameInfo) => (
            <tr key={gameInfo.id}>
              <td>
                <a
                  href={`https://store.steampowered.com/app/${gameInfo.id}`}
                  target="_blank"
                >
                  {gameInfo.name}
                </a>
              </td>
              <td>{gameInfo.price}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default GamesList;
