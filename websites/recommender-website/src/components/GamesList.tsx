import React, { useEffect, useState } from "react";
import { getDocs } from "firebase/firestore";
import { getInteractionsCollection } from "../firebase";
import { fetchGameInfo, GameInfo } from "./GetGameDetails";
import "./GamesList.css";
import MUIDataTable from "mui-datatables";
import { MUIDataTableOptions } from "mui-datatables";
import { createTheme, ThemeProvider } from "@mui/material";

interface GamesListProps {
  userID: string;
}

interface Interaction {
  game_id: number;
  user_liked: boolean;
  timestamp: number;
}

interface RowInfo {
  gameInfo: GameInfo;
  interaction: Interaction;
}

const GamesList: React.FC<GamesListProps> = ({ userID }) => {
  const [rowInfo, setRowInfo] = useState<RowInfo[] | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    (async () => {
      const querySnapshot = await getDocs(getInteractionsCollection());
      const interactions = querySnapshot.docs.map(
        (doc) => doc.data() as Interaction
      );
      const rowPromises = interactions.map((interaction) =>
        fetchGameInfo(interaction.game_id).then((gameInfo) => ({
          gameInfo,
          interaction,
        }))
      );
      const rowInfo = await Promise.all(rowPromises);

      setRowInfo(rowInfo);
    })();
    return () => {
      controller.abort();
    };
  }, [userID]);

  if (rowInfo === null) {
    return "Loading...";
  }

  const formatUnixTimestamp = (unixTimestamp: number): string => {
    const date = new Date(unixTimestamp * 1000);
    return date.toLocaleString("en-US", { timeZone: "UTC" });
  };

  // TODO Add useful filter options for numeric.
  // TODO Make the dropdown pull the screen from recommendation page.
  // TODO Fix visual bugs on mobile.
  const columns = [
    {
      name: "interaction.timestamp",
      label: "Time",
      options: {
        customBodyRender: (value: number) => formatUnixTimestamp(value),
      },
    },
    {
      name: "gameInfo.name",
      label: "Game Name",
      options: {
        filter: false,
        sort: false,
        customBodyRender: (value: string, tableMeta: any) => (
          <u>
            <a
              href={`https://store.steampowered.com/app/${tableMeta.rowData[7]}`}
              target="_blank"
            >
              {value}
            </a>
          </u>
        ),
      },
    },
    {
      name: "interaction.user_liked",
      label: "User Liked",
      options: {
        customBodyRender: (value: boolean) => (
          <span>{value ? "\u2714" : "\u274C"}</span>
        ),
      },
    },
    { name: "gameInfo.price", label: "Price" },
    { name: "gameInfo.numReviews", label: "Num Reviews" },
    { name: "gameInfo.avgReviewScore", label: "Review Score" },
    {
      name: "gameInfo.description",
      label: "Description",
      options: { filter: false, sort: false, display: false },
    },
    {
      name: "gameInfo.id",
      label: "ID",
      options: { filter: false, sort: false, display: false },
    },
  ];

  const theme = createTheme({
    palette: {
      mode: "dark",
    },
  });

  const options: MUIDataTableOptions = {
    enableNestedDataAccess: ".",
    selectableRows: "none",
    expandableRows: true,
    renderExpandableRow: (rowData: string[]) => {
      const colSpan = rowData.length + 1;
      return (
        <tr>
          <td colSpan={colSpan}>
            <p>{rowData[6]}</p>
          </td>
        </tr>
      );
    },
  };

  return (
    <ThemeProvider theme={theme}>
      <div style={{ height: "100%", width: "100%" }}>
        <MUIDataTable
          title="Interactions"
          data={rowInfo}
          columns={columns}
          options={options}
        />
      </div>
    </ThemeProvider>
  );
};

export default GamesList;
