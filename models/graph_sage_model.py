from models.base_model import BaseGameRecommendationModel, SAVED_MODELS_PATH
import torch
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from torch_geometric.data import Data

import os

class GraphSAGE(torch.nn.Module, BaseGameRecommendationModel):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_epochs=20,
        learning_rate=0.01,
        regularization=0.05,
    ):
        super().__init__()
        self.num_layers = len(hidden_channels)  # Assuming hidden_channels is a list
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels[0]))
        for i in range(1, self.num_layers):
            self.convs.append(SAGEConv(hidden_channels[i - 1], hidden_channels[i]))
        self.final_conv = SAGEConv(hidden_channels[-1], out_channels)
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.regularization = regularization

    def forward(self, x, edge_index):
        print(f"Input x shape: {x.shape}")
        print(f"Input edge_index shape: {edge_index.shape}")
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = torch.relu(x)
            print(f"Shape after layer {i}: {x.shape}")
        x = self.final_conv(x, edge_index)
        print(f"Output shape: {x.shape}")
        return x

    
    def generate_embeddings(self):
        # if isinstance(data, bool):
        #     raise ValueError("Expected a PyTorch Geometric Data object, got a boolean value.")
        with torch.no_grad():  # Disable gradient computation
            embeddings = self.forward(self.data.x, self.data.edge_index)
        return embeddings


    def name(self):
        return "graphsage"

    def train(self, data, user_ids, game_ids, debug=False):
        # Add the node_id_to_index mapping to the data object
        self.node_id_to_index = {node_id: i for i, node_id in enumerate(list(set(user_ids + game_ids)))}
        self.data = data
        self.user_ids = user_ids
        self.game_ids = game_ids
        data.node_id_to_index = self.node_id_to_index

        if debug:
            assert isinstance(data, Data), "data must be a PyTorch Geometric Data object"
            print(f"data type: {type(data)}")
            print(f"Has attribute 'x': {'x' in dir(data)}")
            print(f"Has attribute 'edge_index': {'edge_index' in dir(data)}")
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            embeddings = self.forward(data.x, data.edge_index)
            # Define a loss function, e.g., MSE between embeddings and some target
            loss = torch.nn.functional.mse_loss(embeddings, torch.rand_like(embeddings))
            loss.backward()
            optimizer.step()
            if debug:
                print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    def get_score_between_user_and_game(self, user_id, game_id, data):
        print("Data type in method:", type(data))
        embeddings = self.generate_embeddings()
        print("Embeddings shape:", embeddings.shape)
        # Find the indices of the user and game in the data
        user_index = data.node_id_to_index[user_id]
        game_index = data.node_id_to_index[game_id]
        user_embedding = embeddings[user_index]
        game_embedding = embeddings[game_index]
        score = torch.dot(user_embedding, game_embedding)
        return score.item()

    def get_scores_between_users_and_games(self, user_id, game_id):
        return None

    def score_and_predict_n_games_for_user(self, user_id, N=None, should_sort=True):
        embeddings = self.generate_embeddings()
        user_index = self.node_id_to_index[user_id]

        scores = []
        for game_id in self.node_id_to_index.keys():
            if game_id in self.game_ids:  # Assuming you have a list of game_ids
                game_index = self.node_id_to_index[game_id]
                user_embedding = embeddings[user_index]
                game_embedding = embeddings[game_index]
                score = torch.dot(user_embedding, game_embedding).item()
                scores.append((game_id, score))

        if should_sort:
            scores.sort(key=lambda x: x[1], reverse=True)

        if N is not None:
            return scores[:N]

        return scores

    def save(self, file_name, overwrite=False):
        file_path = os.path.join(SAVED_MODELS_PATH, file_name + ".pt")
        if not overwrite and os.path.exists(file_path):
            raise FileExistsError(
                f"Tried to save to a file that already exists {file_name} without allowing for overwrite."
            )
        torch.save(self.state_dict(), file_path)

    def _load(self, folder_path, file_name):
        file_path = os.path.join(folder_path, file_name + ".pt")
        self.load_state_dict(torch.load(file_path))

    def _fine_tune(
        self,
        user_id,
        new_user_games_df,
        new_interactions_df,
        all_user_games_df,
        all_interactions_df,
    ):
        # Update user embedding based on new interactions
        new_user_embedding = torch.mean(
            self.game_embeddings[new_user_games_df["game_id"]], dim=0
        )
        self.user_embeddings[user_id] = new_user_embedding

        # Update game embeddings based on new interactions
        for game_id in new_interactions_df["game_id"]:
            new_game_embedding = torch.mean(
                self.user_embeddings[
                    new_interactions_df[new_interactions_df["game_id"] == game_id][
                        "user_id"
                    ]
                ],
                dim=0,
            )
            self.game_embeddings[game_id] = new_game_embedding
