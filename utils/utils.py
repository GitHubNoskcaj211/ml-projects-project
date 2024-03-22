import numpy as np
import pandas as pd

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

def print_game_name_and_scores(games_df, scores):
    game_ids = [game_id for game_id, score in scores]
    score_values = [score for game_id, score in scores]
    selected_games = games_df[games_df['id'].isin(game_ids)].copy()
    selected_games = selected_games.assign(score=score_values)
    print(selected_games)