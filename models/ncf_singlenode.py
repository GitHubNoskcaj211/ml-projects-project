import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from base_model import SAVED_NN_PATH
import random
from matplotlib import pyplot as plt

# In depth explanation here: https://github.com/recommenders-team/recommenders/blob/main/examples/02_model_collaborative_filtering/ncf_deep_dive.ipynb
class NCF(nn.Module):
    def __init__(
        self,
        num_users,
        num_games,
        model_type="ncf",
        embedding_size=100,
        mlp_hidden_layer_sizes=[16, 8, 4],
        num_epochs=50,
        batch_percent=0.1,
        learning_rate=5e-3,
        output_size=1,
        seed=None,
    ):
        super(NCF, self).__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed

        self.num_users = num_users
        self.num_games = num_games
        self.model_type = model_type.lower()
        self.embedding_size = embedding_size
        self.mlp_hidden_layer_sizes = [2 * embedding_size, *mlp_hidden_layer_sizes]
        self.num_epochs = num_epochs
        self.batch_percent = batch_percent
        self.learning_rate = learning_rate
        self.output_size = output_size

        # check model type
        # generalized collaborative filter, multi layer perceptron, neural collaborative filter
        model_options = ["gcf", "mlp", "ncf"]
        if self.model_type not in model_options:
            raise ValueError(
                "Wrong model type, please select one of this list: {}".format(
                    model_options
                )
            )

        self.ncf_layer_size = embedding_size + self.mlp_hidden_layer_sizes[-1]
        self._create_model()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        self.gcf = self.model_type == 'gcf' or self.model_type == 'ncf'
        self.mlp = self.model_type == 'mlp' or self.model_type == 'ncf'
        self.neurmf = self.model_type == 'ncf'

        print('Total Learnable Parameters:', sum(p.numel() for p in self.parameters() if p.requires_grad))

    def _create_model(self):
        if self.gcf:
            self.embedding_gcf_user = nn.Embedding(self.num_users, self.embedding_size)
            self.embedding_gcf_game = nn.Embedding(self.num_games, self.embedding_size)

        if self.mlp:
            self.embedding_mlp_user = nn.Embedding(
                self.num_users, self.embedding_size
            )
            self.embedding_mlp_game = nn.Embedding(
                self.num_games, self.embedding_size
            )
            self.mlp_layers = nn.ModuleList()
            for layer1, layer2 in zip(self.mlp_hidden_layer_sizes[:-1], self.mlp_hidden_layer_sizes[1:]):
                self.mlp_layers.append(nn.Linear(layer1, layer2))
                self.mlp_layers.append(nn.ReLU())

        if self.neurmf:
            self.ncf_fc = nn.Linear(self.ncf_layer_size, self.output_size)
        else:
            self.ncf_fc = nn.Linear(self.embedding_size, self.output_size)

        self.loss_fn = nn.MSELoss()

    def forward(self, user_index, game_index):
        if self.gcf:
            gcf_user = self.embedding_gcf_user(user_index)
            gcf_game = self.embedding_gcf_game(game_index)
            gcf_vector = gcf_user * gcf_game

        if self.mlp:
            mlp_user = self.embedding_mlp_user(user_index)
            mlp_game = self.embedding_mlp_game(game_index)
            mlp_vector = torch.cat([mlp_user, mlp_game], dim=1)
            for layer in self.mlp_layers:
                mlp_vector = layer(mlp_vector)

        if self.gcf:
            ncf_vector = gcf_vector
        elif self.mlp:
            ncf_vector = mlp_vector
        else:
            ncf_vector = torch.cat([gcf_vector, mlp_vector], dim=1)

        output = self.ncf_fc(ncf_vector)
        return output.squeeze()

    def train(self, user_indices, game_indices, labels, debug=False):
        assert len(user_indices) == len(game_indices) and len(game_indices) == labels.shape[0], 'Inconsistent number of data rows'
        batch_size = int(len(user_indices) * self.batch_percent) + 1

        train_loss = []
        for epoch_count in tqdm(range(self.num_epochs), desc='Training'):
            epoch_loss = []
            indices = np.random.permutation(len(user_indices))
            for batch_start in range(0, len(user_indices), batch_size):
                batch_end = min(batch_start + batch_size, len(user_indices))
                batch_indices = indices[batch_start:batch_end]

                batched_users, batched_games, batched_labels = user_indices[batch_indices], game_indices[batch_indices], labels[batch_indices]
                predictions = self.forward(batched_users, batched_games)
                loss = self.loss_fn(predictions, batched_labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss_value = loss.item()
                epoch_loss.append()
            if debug:
                print(epoch_count, batch_loss_value)
        if debug:
            plt.plot(range(self.num_epochs), train_loss)
            plt.title('Mean Abs Error vs Epoch')

    def save(self, file_name, overwrite=False):
        assert not os.path.isfile(SAVED_NN_PATH + file_name + '.pth') or overwrite, f'Tried to save to a file that already exists {file_name} without allowing for overwrite.'
        torch.save(self.state_dict(), os.path.join(SAVED_NN_PATH, file_name + '.pth'))

    def load(self, file_name):
        self.load_state_dict(torch.load(os.path.join(SAVED_NN_PATH, file_name + '.pth')))

    def predict(self, user_index, game_index, is_list=False):
        """Predict function of this trained model

        Args:
            user_index (list or element of list): user_id or user_id list
            game_index (list or element of list): game_id or game_id list
            is_list (bool): if true, the input is list type
                noting that list-wise type prediction is faster than element-wise's.

        Returns:
            list or float: A list of predicted rating or predicted rating score.
        """
        if is_list:
            output = self.forward(user_index, game_index)
            return list(output.detach().numpy())

        else:
            output = self.forward(user_index, game_index)
            return float(output.game())
