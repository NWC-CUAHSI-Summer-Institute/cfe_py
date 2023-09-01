import numpy as np

def calculate_nse(modeled_data, observed_data):
    mean_observed = np.mean(observed_data)
    numerator = np.sum(np.power(observed_data - modeled_data, 2))
    denominator = np.sum(np.power(observed_data - mean_observed, 2))
    return 1 - np.divide(numerator, denominator)