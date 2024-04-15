from models.base_model import BaseGameRecommendationModel, SAVED_MODELS_PATH
import torch
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from torch_geometric.data import Data
from tqdm import tqdm

import os

class GraphSAGE(torch.nn.Module, BaseGameRecommendationModel):
    def __init__(
        self,
        hidden_channels,
        out_channels,
        num_epochs=20,
        learning_rate=0.01,
        regularization=0.05,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.regularization = regularization

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = torch.relu(x)
        x = self.final_conv(x, edge_index)
        return x

    
    def generate_embeddings(self):
        # if isinstance(data, bool):
        #     raise ValueError("Expected a PyTorch Geometric Data object, got a boolean value.")
        with torch.no_grad():  # Disable gradient computation
            embeddings = self.forward(self.data.x, self.data.edge_index)
        return embeddings


    def name(self):
        return "graphsage"

    def _train(self, debug=False):
        # Add the node_id_to_index mapping to the data object     
        if debug:
            assert isinstance(self.data, Data), "data must be a PyTorch Geometric Data object"
            print(f"data type: {type(self.data)}")
            print(f"Has attribute 'x': {'x' in dir(self.data)}")
            print(f"Has attribute 'edge_index': {'edge_index' in dir(self.data)}")
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        for epoch in tqdm(range(self.num_epochs), desc='Training'):
            optimizer.zero_grad()
            embeddings = self.forward(self.data.x, self.data.edge_index)
            # Define a loss function, e.g., MSE between embeddings and some target
            loss = torch.nn.functional.mse_loss(embeddings, torch.rand_like(embeddings))
            loss.backward()
            optimizer.step()
            if debug:
                print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    
    def train(self, debug=False, user_node_ids=None):
        # TODO train on downloaded interactions
        assert self.data_loader.cache_local_dataset, 'Method requires full load.'
        self.game_ids = self.data_loader.get_game_node_ids()
        self.user_ids = user_node_ids if user_node_ids is not None else self.data_loader.get_user_node_ids()
        self.node_ids = list(set(self.user_ids + self.game_ids))
        self.node_id_to_index = {node_id: i for i, node_id in enumerate(self.node_ids)}
        train_users_games_df = self.data_loader.users_games_df[self.data_loader.users_games_df['data_split'] == 'train']
        edges = [(user, game) for user, game in zip(train_users_games_df['user_id'].tolist(), train_users_games_df['game_id'].tolist())]
        edge_index_data = [(self.node_id_to_index[edge[0]], self.node_id_to_index[edge[1]]) for edge in edges]
        edge_index = torch.tensor(edge_index_data, dtype=torch.long).t().contiguous()
        x = torch.arange(len(self.node_ids), dtype=torch.float).view(-1, 1)
        self.data = Data(x=x, edge_index=edge_index)   
        self.data.node_id_to_index = self.node_id_to_index

        self.in_channels = self.data.x.size(1)
        self.num_layers = len(self.hidden_channels)  # Assuming self.hidden_channels is a list
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(self.in_channels, self.hidden_channels[0]))
        for i in range(1, self.num_layers):
            self.convs.append(SAGEConv(self.hidden_channels[i - 1], self.hidden_channels[i]))
        self.final_conv = SAGEConv(self.hidden_channels[-1], self.out_channels)
        self._train(debug)
        self.embeddings = self.generate_embeddings()

    def get_score_between_user_and_game(self, user_id, game_id, data):
        return None
        # print("Data type in method:", type(data))
        # embeddings = self.generate_embeddings()
        # print("Embeddings shape:", embeddings.shape)
        # # Find the indices of the user and game in the data
        # user_index = data.node_id_to_index[user_id]
        # game_index = data.node_id_to_index[game_id]
        # user_embedding = embeddings[user_index]
        # game_embedding = embeddings[game_index]
        # score = torch.dot(user_embedding, game_embedding)
        # return score.item()

    def get_scores_between_users_and_games(self, user_id, game_id):
        return None

    def score_and_predict_n_games_for_user(self, user_id, N=None, should_sort=True, games_to_include=[]):
        games_to_filter_out = self.data_loader.get_all_game_ids_for_user(user_id)
        user_index = self.node_id_to_index[user_id]

        game_indices = [self.node_id_to_index[game_id] for game_id in self.game_ids]
        user_embedding = self.embeddings[user_index]
        game_embeddings = self.embeddings[game_indices]
        scores = (game_embeddings * user_embedding).sum(dim = 1).tolist()
        scores = [(game_id, score) for game_id, score in zip(self.game_ids, scores)]
        
        return self.select_scores(scores, N = N, should_sort=should_sort, games_to_filter_out=games_to_filter_out, games_to_include=games_to_include)

    def save(self, file_name, overwrite=False):
        return
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
        debug=False
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
