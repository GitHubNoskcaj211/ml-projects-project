def linear_transformation(number, start_domain, end_domain, start_range, end_range):
    if (end_domain - start_domain) == 0:
        return end_range
    slope = (end_range - start_range) / (end_domain - start_domain)
    intercept = start_range - slope * start_domain
    transformed_number = slope * number + intercept
    return transformed_number

def gaussian_transformation(number, old_mean, old_std_dev, new_mean, new_std_dev):
    if old_std_dev == 0:
        return new_mean
    return (number - old_mean) / old_std_dev * new_std_dev + new_mean