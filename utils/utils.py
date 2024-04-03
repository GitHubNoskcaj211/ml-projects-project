import numpy as np
import pandas as pd
from pprint import pprint

def linear_transformation(numbers, start_domain, end_domain, start_range, end_range):
    if (end_domain - start_domain) == 0:
        if isinstance(numbers, (int, float)):
            return end_range
        elif isinstance(numbers, np.ndarray):
            return np.full_like(numbers, end_range)
        elif isinstance(numbers, pd.Series):
            return pd.Series(np.full_like(numbers.values, end_range), index=numbers.index)
        else:
            raise ValueError("Input must be a number or a numpy array.")
    slope = (end_range - start_range) / (end_domain - start_domain)
    intercept = start_range - slope * start_domain
    transformed_numbers = slope * numbers + intercept
    return transformed_numbers

def gaussian_transformation(numbers, old_mean, old_std_dev, new_mean, new_std_dev):
    if old_std_dev == 0 or old_std_dev == np.nan or pd.isna(old_std_dev):
        if isinstance(numbers, (int, float)):
            return new_mean
        elif isinstance(numbers, np.ndarray):
            return np.full_like(numbers, new_mean)
        elif isinstance(numbers, pd.Series):
            return pd.Series(np.full_like(numbers.values, new_mean), index=numbers.index)
        else:
            raise ValueError("Input must be a number or a numpy array.")
    return (numbers - old_mean) / old_std_dev * new_std_dev + new_mean

def get_numeric_dataframe_columns(df, columns_to_remove=[]):
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    numeric_columns = list(set(numeric_columns) - set(columns_to_remove))
    return df[numeric_columns]

def get_game_name_and_scores(data_loader, scores):
    game_ids = [game_id for game_id, score in scores]
    score_values = [score for game_id, score in scores]
    selected_games = [data_loader.get_game_information(game_id) for game_id in game_ids]
    print(selected_games)
    selected_games = pd.DataFrame(selected_games)[['id', 'name']]
    selected_games = selected_games.assign(score=score_values)
    return selected_games