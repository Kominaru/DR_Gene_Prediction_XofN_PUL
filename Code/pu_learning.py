
import numpy as np
from Code.data_processing import load_data
from sklearn.metrics import pairwise_distances

def pairwise_jaccard_measures(x, y):
    """
    Compute the Jaccard Measures for the given Positive Unlabelled dataset.
    Given a dataset with |P| positive samples and |U| unlabelled samples, this
    method returns a |U|x(|P|+|U|) matrix containing the Jaccard Measures between
    each unlabelled sample and the rest of the samples.

    Parameters:
    - x: array-like of shape (n_samples, n_features)
        The dataset.
    - y: array-like of shape (n_samples,). Used to identify the positive samples.

    Returns:
    - jaccard_measures: array-like of shape (|U|, n_samples), where |U| is the number of unlabelled samples.
        The Jaccard Measures.
    """

    intersection = x @ x.T
    x = np.sum(x, axis=1)
    union = np.squeeze(x[:, None] + x.T) - intersection

    return intersection / union

def parwise_jaccard_measures_2(x, y):
    # Compute the Jaccard Measures using sklearn's pairwise_distances

    
    jac = pairwise_distances(x, metric="jaccard")
    return jac
    
x, y = load_data(f"./Data/Datasets/PathDIP.csv")
x = x.to_numpy()

# Compare the speed of both methods
import timeit

print(timeit.timeit(lambda: pairwise_jaccard_measures(x, y), number=100))
x = x.astype(bool)
print(timeit.timeit(lambda: parwise_jaccard_measures_2(x, y), number=100))

