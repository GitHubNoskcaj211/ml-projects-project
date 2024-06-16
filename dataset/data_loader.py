from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from enum import Enum
import pandas as pd
import os
import pickle
from ast import literal_eval
from utils.utils import linear_transformation, gaussian_transformation
from utils.firestore import DatabaseClient
import sqlite3

from dataset.scrape.serialization import deserialize_user, deserialize_users_games, deserialize_game

SAVED_DATA_LOADER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_data_loader_parameters/')
PUBLISHED_MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models/published_recommendation_models/')

USERS_GAMES_SCHEMA = {
            'user_id': 'int64',
            'game_id': 'int64',
            # 'playtime_2weeks': 'int64',
            'playtime_forever': 'int64',
            'source': 'string',
        }

INTERACTIONS_SCHEMA = {
                  'user_id': 'int64',
                  'game_id': 'int64',
                  'user_liked': 'bool',
                  'time_spent': 'float64',
                  'source': 'string',
                  'timestamp': 'int64',
                 }

GAME_SCHEMA = {
            'id': 'int64',
            'name': 'string',
            'numReviews': 'int64',
            'avgReviewScore': 'int64',
            'price': 'float64',
            'numFollowers': 'int64',
        }

USER_SCHEMA = {
            'id': 'int64',
        }

FRIENDS_SCHEMA = {
            'user1': 'int64',
            'user2': 'int64',
        }

LOCAL_DATA_SOURCE = 'local'
EXTERNAL_DATA_SOURCE = 'external'

DATA_FILES_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data_files/')
USERS_FILENAME = 'users.csv'
GAMES_FILENAME = 'games.csv'
USERS_GAMES_FILENAME = 'users_games.csv'
FRIENDS_FILENAME = 'friends.csv'

def get_games_for_user(data_loader, user):
    assert data_loader.cache_local_dataset, 'Method requires full load.'
    return data_loader.users_games_df[data_loader.users_games_df['user_id'] == user]

def read_log_file(file_path):
    result = []
    with open(file_path, 'r') as file:
        for line in file:
            int_list = list(map(int, line.split()))
            result.append(int_list)
    return result

class EmbeddingType(Enum):
    IDENTITY = 0
    CATEGORICAL = 1
    ONE_HOT = 2
    SUM = 3

# TODO FIX THESE NEXT
class Normalizer(ABC):
    @abstractmethod
    def normalize(self, network):
        pass

class LinearNormalizer(Normalizer):
    def __init__(self, min_score, max_score):
        self.min_score = min_score
        self.max_score = max_score
    
    def normalize(self, df):
        df['score'] = df.groupby('user_id')['score'].transform(lambda scores: linear_transformation(scores, scores.min(), scores.max(), self.min_score, self.max_score))

class GaussianNormalizer(Normalizer):
    def __init__(self, new_mean, new_std):
        self.new_mean = new_mean
        self.new_std = new_std
    
    def normalize(self, df):
        df['score'] = df.groupby('user_id')['score'].transform(lambda scores: gaussian_transformation(scores, scores.mean(), scores.std(), self.new_mean, self.new_std))

class PercentileNormalizer(Normalizer):
    def normalize(self, df):
        df['score'] = df.groupby('user_id')['score'].transform(lambda scores: (scores.rank(pct=True)))

def constant_users_games_edge_scoring_function(users_games_df):
    return pd.Series([1] * len(users_games_df))

def playtime_forever_users_games_edge_scoring_function(users_games_df):
    return users_games_df['playtime_forever']

def liked_interactions_edge_scoring_function(interaction_df):
    # 1 when liked and -1 when not liked
    return interaction_df['user_liked'] * 2 - 1

def never_remove_edge(users_games_df):
    return pd.Series([False] * len(users_games_df))

def remove_zero_playtime_edge(users_games_df):
    return users_games_df['playtime_forever'] == 0

def add_categorical_embedding(df, category_base_name, nested_values):
    categories = set([value for lst in nested_values for value in lst])
    new_columns = []
    for category in categories:
        new_column = nested_values.apply(lambda row_values: 1.0 if category in row_values else 0.0)
        new_columns.append(new_column)
    new_df = pd.concat(new_columns, axis=1)
    new_df.columns = [category_base_name + category for category in categories]
    df = pd.concat([df, new_df], axis=1)
    return df

# Default is a network with game and user nodes (hashed with ids) and edges between users and games. All options are in init.
class DataLoader():
    def __init__(self, users_games_edge_scoring_function = constant_users_games_edge_scoring_function, interactions_edge_scoring_function = liked_interactions_edge_scoring_function, users_games_edge_score_normalizers = [], interactions_score_normalizers = [], user_embeddings = [], game_embeddings = [], user_game_edge_embeddings = [], friend_friend_edge_embeddings = [], snowballs_ids = [], num_users_to_load_per_snowball = None, remove_users_games_edges_function = never_remove_edge, cache_local_dataset = False, get_local = True, get_external_database = False):
        super().__init__()
        self.users_games_edge_scoring_function = users_games_edge_scoring_function
        self.interactions_edge_scoring_function = interactions_edge_scoring_function
        self.users_games_edge_score_normalizers = users_games_edge_score_normalizers
        self.interactions_score_normalizers = interactions_score_normalizers
        self.user_embeddings = user_embeddings
        self.game_embeddings = game_embeddings
        self.user_game_edge_embeddings = user_game_edge_embeddings
        self.friend_friend_edge_embeddings = friend_friend_edge_embeddings
        self.snowball_ids = snowballs_ids
        self.num_users_to_load_per_snowball = num_users_to_load_per_snowball

        self.remove_users_games_edges_function = remove_users_games_edges_function

        self.cache_local_dataset = cache_local_dataset
        if self.cache_local_dataset:
            self.load_local_dataset()

        self.get_local = get_local
        self.get_external_database = get_external_database
        if self.get_external_database:
            self.database_client = DatabaseClient()

    def run_local_database_query(self, query):
        database = sqlite3.connect(f'{DATA_FILES_DIRECTORY}global_database.db')
        result = pd.read_sql_query(query, database)
        database.close()
        return result

    def get_users_games_df_for_user(self, user_id, get_local=True, get_external_database=True, preprocess=False):
        df = pd.DataFrame(columns=USERS_GAMES_SCHEMA.keys()).astype(USERS_GAMES_SCHEMA)
        if self.get_local and get_local:
            if self.cache_local_dataset:
                users_games_df_for_user = self.users_games_df_grouped_by_user.get_group(user_id)
                if self.train_or_eval_mode:
                    new = users_games_df_for_user[users_games_df_for_user['data_split'] != 'test']
                    df = pd.concat([df, new])
                else:
                    df = pd.concat([df, users_games_df_for_user])
            else:
                new_df = deserialize_users_games(user_id)
                if new_df is not None:
                    new_df['source'] = LOCAL_DATA_SOURCE
                    df = pd.concat([df, new_df])
        if self.get_external_database and get_external_database:
            db_data = self.database_client.users_games_ref.document(str(user_id)).get()
            if db_data.exists:
                db_data = db_data.to_dict()["games"]
                db_data = pd.DataFrame(db_data)
                db_data['source'] = EXTERNAL_DATA_SOURCE
                df = pd.concat([db_data, df])
        df.drop_duplicates(subset=["game_id"], keep="first", inplace=True)
        if preprocess:
            df = self.preprocess_users_games_df(df)
        df = df.astype(USERS_GAMES_SCHEMA)
        return df
    
    def get_interactions_df_for_user(self, user_id, get_local=True, get_external_database=True, preprocess=False):
        df = pd.DataFrame(columns=INTERACTIONS_SCHEMA.keys()).astype(INTERACTIONS_SCHEMA)
        if self.get_local and get_local:
            if self.cache_local_dataset:
                pass
                # TODO
                # df = pd.concat([df, self.users_games_df[self.users_games_df['user_id'] == user_id]])
            else:
                pass
                # TODO
                # query = f"SELECT * FROM users_games WHERE user_id = {user_id}"
                # df = pd.concat([df, self.run_local_database_query(query)])
        
        if self.get_external_database and get_external_database:
            recommendation_interactions = self.database_client.interactions_ref.document('data').collection(str(user_id)).get()
            if recommendation_interactions:
                interactions = [interaction_document.to_dict() for interaction_document in recommendation_interactions]
                db_data = pd.DataFrame(interactions)
                db_data['source'] = EXTERNAL_DATA_SOURCE
                df = pd.concat([db_data, df])
        df.drop_duplicates(subset=["game_id"], keep="first", inplace=True)
        if preprocess:
            df = self.preprocess_interactions_df(df)
        df = df.astype(INTERACTIONS_SCHEMA)
        return df
    
    def get_all_game_ids_for_user(self, user_id):
        user_games_df = self.get_users_games_df_for_user(user_id)
        interactions_df = self.get_interactions_df_for_user(user_id)
        return user_games_df['game_id'].to_list() + interactions_df['game_id'].to_list()

    def get_game_information(self, game_id):
        output = None
        if self.get_local:
            if self.cache_local_dataset:
                output = self.games_df[self.games_df['id'] == game_id].to_dict('records')
                output = output[0] if len(output) > 0 else None
            else:
                output = deserialize_game(game_id)
        if self.get_external_database and output is None:
            info = self.database_client.games_ref.document(str(game_id)).get()
            if info.exists:
                output = info.to_dict()
        return output

    def user_exists(self, user_id):
        if self.get_local:
            if self.cache_local_dataset:
                if not self.users_df[self.users_df['id'] == user_id].empty:
                    return True
            else:
                if deserialize_user(user_id):
                    return True
        if self.get_external_database:
            if self.database_client.users_games_ref.document(str(user_id)).get().exists:
                return True
        return False
    
    def get_user_node_ids(self):
        assert self.cache_local_dataset, 'Method requires full load.'
        return self.users_df['id'].unique().tolist()
    
    def get_game_node_ids(self):
        assert self.cache_local_dataset, 'Method requires full load.'
        return self.games_df['id'].unique().tolist()

    def load_random_edge_train_test_split(self, train_percentage=0.9, test_percentage=0.1, seed=0):
        assert self.cache_local_dataset, 'Method requires full load.'
        assert train_percentage + test_percentage <= 1
        self.train_or_eval_mode = True

        train_edges, test_edges = train_test_split(self.users_games_df.index, test_size=test_percentage, train_size=train_percentage, random_state=seed)
        self.users_games_df['data_split'] = 'none'
        self.users_games_df.loc[train_edges, 'data_split'] = 'train'
        self.users_games_df.loc[test_edges, 'data_split'] = 'test'
    
    def load_stratified_user_id_edge_train_test_split(self, train_percentage=0.9, test_percentage=0.1, seed=0):
        assert self.cache_local_dataset, 'Method requires full load.'
        assert train_percentage + test_percentage <= 1
        self.train_or_eval_mode = True

        user_counts = self.users_games_df['user_id'].value_counts()
        valid_users = user_counts[user_counts > 1].index
        strat_classes = self.users_games_df.apply(lambda row: row['user_id'] if row['user_id'] in valid_users else -1, axis=1)

        train_edges, test_edges = train_test_split(self.users_games_df.index, test_size=test_percentage, train_size=train_percentage, random_state=seed, stratify=strat_classes)
        self.users_games_df['data_split'] = 'none'
        self.users_games_df.loc[train_edges, 'data_split'] = 'train'
        self.users_games_df.loc[test_edges, 'data_split'] = 'test'

    def load_random_edge_train_tune_test_split(self, train_percentage=0.8, fine_tune_percentage=0.1, test_percentage=0.1, seed=0):
        assert self.cache_local_dataset, 'Method requires full load.'
        assert train_percentage + fine_tune_percentage + test_percentage <= 1
        self.train_or_eval_mode = True

        train_edges, test_edges = train_test_split(self.users_games_df.index, test_size=test_percentage, train_size=train_percentage + fine_tune_percentage, random_state=seed)
        sum_fine_tune_train_percentage = train_percentage + fine_tune_percentage
        train_edges, fine_tune_edges = train_test_split(train_edges, test_size=fine_tune_percentage / sum_fine_tune_train_percentage, train_size=train_percentage / sum_fine_tune_train_percentage)
        self.users_games_df['data_split'] = 'none'
        self.users_games_df.loc[train_edges, 'data_split'] = 'train'
        self.users_games_df.loc[test_edges, 'data_split'] = 'test'
        self.users_games_df.loc[fine_tune_edges, 'data_split'] = 'tune'

    def load_random_user_train_tune_test_split(self, train_user_percentage=0.8, test_user_percentage=0.2, fine_tune_edge_percentage=0.5, test_edge_percentage=0.5, seed=0):
        assert self.cache_local_dataset, 'Method requires full load.'
        assert train_user_percentage + test_user_percentage <= 1
        assert fine_tune_edge_percentage + test_edge_percentage <= 1
        self.train_or_eval_mode = True
        
        train_users, test_users = train_test_split(self.users_df['id'].unique(), test_size=test_user_percentage, train_size=train_user_percentage, random_state=seed)
        fine_tune_edges, test_edges = train_test_split(self.users_games_df[self.users_games_df['user_id'].isin(test_users)].index, test_size=test_edge_percentage, train_size=fine_tune_edge_percentage)
        self.users_games_df['data_split'] = 'none'
        self.users_games_df.loc[self.users_games_df['user_id'].isin(train_users), 'data_split'] = 'train'
        self.users_games_df.loc[test_edges, 'data_split'] = 'test'
        self.users_games_df.loc[fine_tune_edges, 'data_split'] = 'tune'

    @classmethod
    def load_from_file(cls, file_name, use_published_models_path=False, load_live_data_loader=False):
        folder_path = PUBLISHED_MODELS_PATH if use_published_models_path else SAVED_DATA_LOADER_PATH
        with open(folder_path + file_name + '.pkl', 'rb') as file:
            parameter_dictionary = pickle.load(file)
            return cls(
                users_games_edge_scoring_function = parameter_dictionary['users_games_edge_scoring_function'],
                interactions_edge_scoring_function = parameter_dictionary['interactions_edge_scoring_function'],
                users_games_edge_score_normalizers = parameter_dictionary['users_games_edge_score_normalizers'],
                interactions_score_normalizers = parameter_dictionary['interactions_score_normalizers'],
                user_embeddings = parameter_dictionary['user_embeddings'],
                game_embeddings = parameter_dictionary['game_embeddings'],
                user_game_edge_embeddings = parameter_dictionary['user_game_edge_embeddings'],
                friend_friend_edge_embeddings = parameter_dictionary['friend_friend_edge_embeddings'],
                snowballs_ids = parameter_dictionary['snowballs_ids'],
                num_users_to_load_per_snowball = parameter_dictionary['num_users_to_load_per_snowball'],
                remove_users_games_edges_function = parameter_dictionary['remove_users_games_edges_function'],
                cache_local_dataset = parameter_dictionary['cache_local_dataset'] and not load_live_data_loader,
                get_local = parameter_dictionary['get_local'] or load_live_data_loader,
                get_external_database = parameter_dictionary['get_external_database'] or load_live_data_loader)
    
    def get_data_loader_parameters(self):
        return {
            'users_games_edge_scoring_function': self.users_games_edge_scoring_function,
            'interactions_edge_scoring_function': self.interactions_edge_scoring_function,
            'users_games_edge_score_normalizers': self.users_games_edge_score_normalizers,
            'interactions_score_normalizers': self.interactions_score_normalizers,
            'user_embeddings': self.user_embeddings,
            'game_embeddings': self.game_embeddings,
            'user_game_edge_embeddings': self.user_game_edge_embeddings,
            'friend_friend_edge_embeddings': self.friend_friend_edge_embeddings,
            'snowballs_ids': self.snowball_ids,
            'num_users_to_load_per_snowball': self.num_users_to_load_per_snowball,
            'remove_users_games_edges_function': self.remove_users_games_edges_function,
            'cache_local_dataset': self.cache_local_dataset,
            'get_local': self.get_local,
            'get_external_database': self.get_external_database,
        }
    
    def save_data_loader_parameters(self, file_name, overwrite=False):
        assert not os.path.isfile(SAVED_DATA_LOADER_PATH + file_name + '.pkl') or overwrite, f'Tried to save to a file that already exists {file_name} without allowing for overwrite.'
        with open(SAVED_DATA_LOADER_PATH + file_name + '.pkl', 'wb') as file:
            pickle.dump(self.get_data_loader_parameters(), file)
    
    def load_local_dataset(self):
        if len(self.snowball_ids) == 0 and self.num_users_to_load_per_snowball is not None:
            self.snowball_ids = [subfolder for subfolder in os.listdir(DATA_FILES_DIRECTORY) if os.path.isdir(os.path.join(DATA_FILES_DIRECTORY, subfolder))]

        global_users_df = pd.read_csv(DATA_FILES_DIRECTORY + USERS_FILENAME, dtype=USER_SCHEMA)
        global_games_df = pd.read_csv(DATA_FILES_DIRECTORY + GAMES_FILENAME, dtype=GAME_SCHEMA, converters={'genres': literal_eval, 'tags': literal_eval})
        global_users_games_df = pd.read_csv(DATA_FILES_DIRECTORY + USERS_GAMES_FILENAME, dtype=USERS_GAMES_SCHEMA)
        global_friends_df = pd.read_csv(DATA_FILES_DIRECTORY + FRIENDS_FILENAME, dtype=FRIENDS_SCHEMA)

        if len(self.snowball_ids) == 0 and self.num_users_to_load_per_snowball is None:
            full_users_df = global_users_df
            full_games_df = global_games_df
            full_users_games_df = global_users_games_df
            full_friends_df = global_friends_df
        else:
            users_dfs = []
            games_dfs = []
            for snowball_id in self.snowball_ids:
                users_snowball_df = pd.read_csv(DATA_FILES_DIRECTORY + snowball_id + '/' + USERS_FILENAME, usecols=['id'], dtype={
                    'id': 'int64',
                })
                if self.num_users_to_load_per_snowball is not None:
                    users_snowball_df = users_snowball_df.iloc[:self.num_users_to_load_per_snowball]
                games_snowball_df = pd.read_csv(DATA_FILES_DIRECTORY + snowball_id + '/' + GAMES_FILENAME, usecols=['id'], dtype={
                    'id': 'int64'
                })
                users_dfs.append(users_snowball_df)
                games_dfs.append(games_snowball_df)

            users_df = pd.concat(users_dfs, ignore_index=True)
            games_df = pd.concat(games_dfs, ignore_index=True)

            full_users_df = global_users_df[global_users_df['id'].isin(users_df['id'])]
            full_games_df = global_games_df[global_games_df['id'].isin(games_df['id'])]
            full_users_games_df = global_users_games_df[
                (global_users_games_df['user_id'].isin(users_df['id'])) &
                (global_users_games_df['game_id'].isin(games_df['id']))
            ]
            full_friends_df = global_friends_df[
                (global_friends_df['user1'].isin(users_df['id'])) |
                (global_friends_df['user2'].isin(users_df['id']))
            ]
        self.train_or_eval_mode = False    
        self.users_df = self.add_user_embeddings(full_users_df, self.user_embeddings)
        self.games_df = self.add_game_embeddings(full_games_df, self.game_embeddings)
        self.users_games_df = self.add_user_game_edge_embeddings(full_users_games_df, self.user_game_edge_embeddings)
        self.users_games_df['source'] = LOCAL_DATA_SOURCE
        self.friends_df = self.add_friend_friend_edge_embeddings(full_friends_df, self.friend_friend_edge_embeddings)

        self.users_games_df = self.preprocess_users_games_df(self.users_games_df)
        self.users_games_df_grouped_by_user = self.users_games_df.groupby('user_id')

    def preprocess_users_games_df(self, users_games_df):
        if self.remove_users_games_edges_function != never_remove_edge:
            users_games_df = self.remove_users_games_edges(users_games_df)
            
        self.score_users_games_edges(users_games_df)
        
        for normalizer in self.users_games_edge_score_normalizers:
            normalizer.normalize(users_games_df)
        
        return users_games_df
    
    def preprocess_interactions_df(self, interactions_df):
        self.score_interactions_edges(interactions_df)

        for normalizer in self.interactions_score_normalizers:
            normalizer.normalize(interactions_df)
        
        return interactions_df

    def add_user_embeddings(self, full_users_df, user_embeddings):
        base_users_df = pd.DataFrame()
        base_users_df['id'] = full_users_df['id']
        for user_embedding in user_embeddings:
            match user_embedding:
                case _:
                    raise NotImplementedError(f'Cannot recognize embedding: {user_embedding}')
        return base_users_df
    
    def add_game_embeddings(self, full_games_df, game_embeddings):
        base_games_df = pd.DataFrame()
        base_games_df['id'] = full_games_df['id']
        for game_embedding in game_embeddings:
            match game_embedding:
                case 'name':
                    base_games_df['name'] = full_games_df['name']
                case 'numReviews':
                    base_games_df['num_reviews'] = full_games_df['numReviews']
                case 'avgReviewScore':
                    base_games_df['avg_review_score'] = full_games_df['avgReviewScore']
                case 'price':
                    base_games_df['price'] = full_games_df['price']
                case 'genres':
                    base_games_df = add_categorical_embedding(base_games_df, 'genre: ', full_games_df['genres'])
                case 'tags':
                    base_games_df = add_categorical_embedding(base_games_df, 'tag: ', full_games_df['tags'])
                case 'numFollowers':
                    base_games_df['num_followers'] = full_games_df['numFollowers']
                case _:
                    raise NotImplementedError(f'Cannot recognize embedding: {game_embedding}')
        return base_games_df

    def add_user_game_edge_embeddings(self, full_users_games_df, user_game_edge_embeddings):
        base_users_games_df = pd.DataFrame()
        base_users_games_df[['user_id', 'game_id']] = full_users_games_df[['user_id', 'game_id']]
        for user_game_edge_embedding in user_game_edge_embeddings:
            match user_game_edge_embedding:
                case 'playtime_forever':
                    base_users_games_df['playtime_forever'] = full_users_games_df['playtime_forever']
                case _:
                    raise NotImplementedError(f'Cannot recognize embedding: {user_game_edge_embedding}')
        return base_users_games_df

    def add_friend_friend_edge_embeddings(self, full_friends_df, friend_friend_edge_embeddings):
        base_friends_df = pd.DataFrame()
        base_friends_df[['user1', 'user2']] = full_friends_df[['user1', 'user2']]
        for friend_friend_edge_embedding in friend_friend_edge_embeddings:
            match friend_friend_edge_embedding:
                case _:
                    raise NotImplementedError(f'Cannot recognize embedding: {friend_friend_edge_embedding}')
        return base_friends_df

    def remove_users_games_edges(self, users_games_df):
        return users_games_df[~self.remove_users_games_edges_function(users_games_df)].copy()

    def score_users_games_edges(self, users_games_df):
        users_games_df['score'] = self.users_games_edge_scoring_function(users_games_df)
        users_games_df['score'] = users_games_df['score'].astype('float64')

    def score_interactions_edges(self, interactions_df):
        interactions_df['score'] = self.interactions_edge_scoring_function(interactions_df)
        interactions_df['score'] = interactions_df['score'].astype('float64')
