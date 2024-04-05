import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.base_model import SAVED_MODELS_PATH, SAVED_NN_PATH
from matplotlib import pyplot as plt
import random
import pandas as pd

# In depth explanation here: https://github.com/recommenders-team/recommenders/blob/main/examples/02_model_collaborative_filtering/ncf_deep_dive.ipynb
class NCF(nn.Module):
    def __init__(self, num_users, num_games, model_type="ncf", embedding_size=100, known_game_embeddings_df=pd.DataFrame(), mlp_hidden_layer_sizes=[16, 8, 4], num_epochs=50, batch_percent=0.1, learning_rate=5e-3, weight_decay=1e-5, seed=None):
        super(NCF, self).__init__()

        self.new_seed(seed)

        self.known_game_embeddings_df = known_game_embeddings_df
        self.known_game_embeddings_tensor = torch.tensor(self.known_game_embeddings_df.values, dtype=torch.float32)
        self.num_known_game_embeddings = len(self.known_game_embeddings_df.columns)

        self.num_users = num_users
        self.num_games = num_games
        self.model_type = model_type.lower()
        self.embedding_size = embedding_size
        self.mlp_hidden_layer_sizes = [2 * embedding_size + self.num_known_game_embeddings, *mlp_hidden_layer_sizes]
        self.num_epochs = num_epochs
        self.batch_percent = batch_percent
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # generalized collaborative filter, multi layer perceptron, neural collaborative filter
        model_options = ["cf", "gcf", "mlp", "ncf"]
        if self.model_type not in model_options:
            raise ValueError(
                "Wrong model type, please select one of this list: {}".format(
                    model_options
                )
            )

        self.cf = self.model_type == 'cf'
        self.gcf = self.model_type == 'gcf' or self.model_type == 'ncf'
        self.mlp = self.model_type == 'mlp' or self.model_type == 'ncf'
        self.ncf = self.model_type == 'ncf'

        self._create_model()

    def new_seed(self, seed=None):
        if seed is None:
            torch.seed()
        else:
            torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed

    def _create_model(self):
        if self.gcf or self.cf:
            self.embedding_gcf_user = nn.Embedding(self.num_users, self.embedding_size)
            self.embedding_gcf_game = nn.Embedding(self.num_games, self.embedding_size)
            self.embedding_gcf_known_game = nn.Embedding.from_pretrained(self.known_game_embeddings_tensor)
            self.embedding_gcf_user_for_known_game = nn.Embedding(self.num_users, self.num_known_game_embeddings)
            
        if self.mlp:
            self.embedding_mlp_user = nn.Embedding(self.num_users, self.embedding_size)
            self.embedding_mlp_game = nn.Embedding(self.num_games, self.embedding_size)
            self.embedding_mlp_known_game = nn.Embedding.from_pretrained(self.known_game_embeddings_tensor)
            self.mlp_layers = nn.ModuleList()
            for layer1, layer2 in zip(self.mlp_hidden_layer_sizes[:-1], self.mlp_hidden_layer_sizes[1:]):
                self.mlp_layers.append(nn.Linear(layer1, layer2))
                self.mlp_layers.append(nn.ReLU())

        if self.ncf:
            self.ncf_fc = nn.Linear(self.embedding_size + self.num_known_game_embeddings + self.mlp_hidden_layer_sizes[-1], 1)
        elif self.gcf:
            self.ncf_fc = nn.Linear(self.embedding_size + self.num_known_game_embeddings, 1)
        elif self.mlp:
            self.ncf_fc = nn.Linear(self.mlp_hidden_layer_sizes[-1], 1)
        
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.loss_fn = nn.L1Loss()

    def forward(self, user_index, game_index):
        if self.cf or self.gcf:
            gcf_vector = torch.cat([self.embedding_gcf_user(user_index) * self.embedding_gcf_game(game_index), self.embedding_gcf_user_for_known_game(user_index) * self.embedding_gcf_known_game(game_index)], dim=1)
            if self.gcf:
                gcf_vector = self.relu(gcf_vector)

        if self.mlp:
            mlp_vector = torch.cat([self.embedding_mlp_user(user_index), self.embedding_mlp_game(game_index), self.embedding_mlp_known_game(game_index)], dim=1)
            for layer in self.mlp_layers:
                mlp_vector = layer(mlp_vector)

        if self.ncf:
            ncf_vector = torch.cat([gcf_vector, mlp_vector], dim=1)
        elif self.cf:
            ncf_vector = gcf_vector
        elif self.gcf:
            ncf_vector = gcf_vector
        elif self.mlp:
            ncf_vector = mlp_vector

        if self.cf:
            output = torch.sum(ncf_vector, dim=-1, keepdim=True)
        else:
            ncf_vector = self.dropout(ncf_vector)
            output = self.ncf_fc(ncf_vector)
        return output

    def train(self, user_indices, game_indices, labels, debug=False, writer=None):
        super(NCF, self).train(True)
        assert len(user_indices) == len(game_indices) and len(game_indices) == labels.shape[0], 'Inconsistent number of data rows'
        for p in self.parameters():
            p.requires_grad_(True)
        if self.gcf or self.cf:
            self.embedding_gcf_known_game.weight.requires_grad_(False)
        if self.mlp:
            self.embedding_mlp_known_game.weight.requires_grad_(False)
        # TODO Fix weight decay for these
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)
        print('Total Learnable Parameters:', sum(p.numel() for p in self.parameters() if p.requires_grad))
        batch_size = int(len(user_indices) * self.batch_percent) + 1

        train_loss = []
        for epoch_count in tqdm(range(self.num_epochs), desc='Training'):
            epoch_loss = []
            indices = np.random.permutation(len(user_indices))
            for batch_start in range(0, len(user_indices), batch_size):
                batch_end = min(batch_start + batch_size, len(user_indices))
                batch_indices = indices[batch_start:batch_end]

                batched_users = user_indices[batch_indices]
                batched_games = game_indices[batch_indices]
                batched_labels = labels[batch_indices]
                predictions = self.forward(batched_users, batched_games)
                loss = self.loss_fn(predictions, batched_labels)
                if writer is not None:
                    writer.add_scalar('Loss/train', loss, epoch_count)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss.append(loss.item())
            average_epoch_loss = sum((value for value in epoch_loss)) / len(epoch_loss)
            train_loss.append(average_epoch_loss)
        if debug:
            plt.plot(range(self.num_epochs), train_loss)
            plt.title('Mean Abs Error vs Epoch')

    def add_new_user(self):
        # Init with average user embedding + normal random around it?
        if self.gcf or self.cf:
            # new_weight = torch.cat([self.embedding_gcf_user.weight, self.embedding_gcf_user.weight[random.randint(0, self.num_users)].reshape(1, -1)])
            new_weight = torch.cat([self.embedding_gcf_user.weight, torch.mean(self.embedding_gcf_user.weight, dim=0, keepdim=True)])
            # new_weight = torch.cat([self.embedding_gcf_user.weight, torch.randn(1, self.embedding_size)])
            self.embedding_gcf_user = nn.Embedding.from_pretrained(new_weight)
            # new_weight = torch.cat([self.embedding_gcf_user_for_known_game.weight, self.embedding_gcf_user_for_known_game.weight[random.randint(0, self.num_users)].reshape(1, -1)])
            new_weight = torch.cat([self.embedding_gcf_user_for_known_game.weight, torch.mean(self.embedding_gcf_user_for_known_game.weight, dim=0, keepdim=True)])
            # new_weight = torch.cat([self.embedding_gcf_user_for_known_game.weight, torch.randn(1, self.num_known_game_embeddings)])
            self.embedding_gcf_user_for_known_game = nn.Embedding.from_pretrained(new_weight)
        if self.mlp:
            # new_weight = torch.cat([self.embedding_mlp_user.weight, self.embedding_mlp_user.weight[random.randint(0, self.num_users)].reshape(1, -1)])
            new_weight = torch.cat([self.embedding_mlp_user.weight, torch.mean(self.embedding_mlp_user.weight, dim=0, keepdim=True)])
            # new_weight = torch.cat([self.embedding_mlp_user.weight, torch.randn(1, self.embedding_size)])
            self.embedding_mlp_user = nn.Embedding.from_pretrained(new_weight)
        self.num_users += 1

    def fine_tune(self, user_index, user_indices, game_indices, labels, num_epochs, learning_rate, weight_decay, debug=False):
        super(NCF, self).train(True)
        assert len(user_indices) == len(game_indices) and len(game_indices) == labels.shape[0], 'Inconsistent number of data rows'
        for p in self.parameters():
            p.requires_grad_(False)
        if self.gcf or self.cf:
            self.embedding_gcf_user.weight.requires_grad_(True)
        if self.mlp:
            self.embedding_mlp_user.weight.requires_grad_(True)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate, weight_decay=0)
        train_loss = []
        weight_decay_indices = torch.LongTensor([user_index])
        for epoch_count in range(num_epochs):
            predictions = self.forward(user_indices, game_indices)
            loss = self.loss_fn(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            with torch.no_grad():
                if self.gcf or self.cf:
                    self.embedding_gcf_user.weight[user_index] = self.embedding_gcf_user.weight[user_index] - weight_decay * self.embedding_gcf_user(weight_decay_indices)
                if self.mlp:
                    self.embedding_mlp_user.weight[user_index] = self.embedding_mlp_user.weight[user_index] - weight_decay * self.embedding_mlp_user(weight_decay_indices)
            optimizer.step()
            train_loss.append(loss.item())
        if debug:
            plt.plot(range(num_epochs), train_loss)
            plt.title('Mean Abs Error vs Epoch')

    def test_loss(self, user_indices, game_indices, labels):
        super(NCF, self).train(False)
        predictions = self.forward(user_indices, game_indices)
        loss = self.loss_fn(predictions, labels)
        return loss.item()

    def save(self, file_name, overwrite=False):
        assert not os.path.isfile(SAVED_MODELS_PATH + SAVED_NN_PATH + file_name + '.pth') or overwrite, f'Tried to save to a file that already exists {file_name} without allowing for overwrite.'
        torch.save(self.state_dict(), os.path.join(SAVED_MODELS_PATH, SAVED_NN_PATH, file_name + '.pth'))

    def load(self, folder_path, file_name):
        self.load_state_dict(torch.load(os.path.join(folder_path, SAVED_NN_PATH, file_name + '.pth')))

    def predict(self, user_index, game_index):
        super(NCF, self).train(False)
        with torch.no_grad():
            output = self.forward(user_index, game_index)
            return output.detach().flatten().tolist()
