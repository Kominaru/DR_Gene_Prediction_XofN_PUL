
import numpy as np
from Code.data_processing import load_data
from sklearn.metrics import pairwise_distances

distances = None

def compute_pairwise_jaccard_measures(x):

    global distances

    x = x.to_numpy().astype(bool)

    distances = pairwise_distances(x, metric="jaccard")
    distances[np.diag_indices_from(distances)] = np.nan
    
def select_reliable_negatives(p, u, k, t):

    global distances

    d_subset = np.concatenate([p, u]).T

    distances_subset = distances.copy()

    distances_subset[:, ~np.isin(np.arange(distances_subset.shape[1]), d_subset)] = np.nan

    # print("Percentage of NaNs in distances_subset:", np.isnan(distances_subset).sum() / distances_subset.size)

    topk = np.argsort(distances_subset, axis=1)[:, :k] # Indices of the k closest genes for each gene

    topk_is_unlabelled = np.isin(topk, u) 
    closest_unlabelled = topk_is_unlabelled[:, 0] # Condition 1: Closest gene is unlabelled
    topk_percent_unlabelled = np.mean(topk_is_unlabelled, axis=1)

    # from matplotlib import pyplot as plt
    # plt.hist(topk_percent_unlabelled[d_subset], bins=[(i-1)/10+1/(10*2) for i in range(12)], color='#CCCCCC', label='Examples where $x_{max\_sim} \in P$', alpha=1)
    # plt.show()
    
    reliable_negatives = np.where((closest_unlabelled & (topk_percent_unlabelled >= t)))[0]
    reliable_negatives = np.setdiff1d(reliable_negatives, p, assume_unique=True)

    print("\t\t\t",f"RN: {len(reliable_negatives)} (P: {len(p)}, U: {len(u)})")

    return reliable_negatives


