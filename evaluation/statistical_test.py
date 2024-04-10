import numpy as np
import pandas as pd

def generator_for_normal_distribution(mean, variance):
    return lambda: np.random.normal(mean, np.sqrt(variance))

def estimate_probability_each_model_is_best(result_dataframe, metric, num_samples=100000, optimization_type='max'):
    means = result_dataframe.loc[(result_dataframe[metric].notna()) & (result_dataframe[f'{metric}_variance'].notna()), metric].tolist()
    variances = result_dataframe.loc[(result_dataframe[metric].notna()) & (result_dataframe[f'{metric}_variance'].notna()), f'{metric}_variance'].tolist()

    if optimization_type == 'max':
        optimization_function = np.argmax
    elif optimization_type == 'min':
        optimization_function = np.argmin
    else:
        raise Exception('Choose valid optimization type.')

    best_counts = [0] * len(means)
    generators = [generator_for_normal_distribution(mean, variance) for mean, variance in zip(means, variances)]
    for ii in range(num_samples):
        generated_values = [generator() for generator in generators]
        best_counts[optimization_function(generated_values)] += 1

    probabilities = [cc / num_samples for cc in best_counts]
    result_dataframe.loc[(result_dataframe[metric].notna()) & (result_dataframe[f'{metric}_variance'].notna()), f'{metric}_best_probability'] = probabilities

def weighted_variance(scores, counts):
    total_tests = sum(counts)
    if total_tests == 0:
        return None, None
    weighted_mean = sum(score * count for score, count in zip(scores, counts)) / total_tests
    if total_tests == 1:
        return weighted_mean, None
    weighted_variance = sum(count * (score - weighted_mean) ** 2 for score, count in zip(scores, counts)) / (total_tests - 1)
    return weighted_mean, weighted_variance