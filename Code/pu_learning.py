
import numpy as np
from code.data_processing import get_data_features
from sklearn.metrics import pairwise_distances
from code.models import get_model, set_class_weights
from skrebate import ReliefF

distances = None

def compute_pairwise_jaccard_measures(x):

    global distances

    x = x.to_numpy().astype(bool)

    distances = pairwise_distances(x, metric="jaccard")
    distances[np.diag_indices_from(distances)] = np.nan

def feature_selection_jaccard(x, y, n_features, selection_method='all', random_state=42, **kwargs):

    x_feat = get_data_features(x)

    if selection_method == 'model':

        model = get_model(kwargs['classifier'], random_state=random_state)

        if kwargs['classifier'] == 'CAT':
            model = set_class_weights(model, kwargs['sampling_method'], y)
            model.fit(x_feat, y, verbose=0)
        else:
            model.fit(x_feat, y)

        feature_importances = model.feature_importances_

        idx = np.argsort(feature_importances)[::-1][:n_features]

        x_feat = x_feat.iloc[:, idx]

        compute_pairwise_jaccard_measures(x_feat)

    if selection_method == "relieff":

        fs = ReliefF(n_features_to_select=n_features, n_neighbors=10, n_jobs=8)

        fs.fit(x_feat.to_numpy(), y)
        x_feat = x_feat.iloc[:, fs.top_features_[:n_features]]

        compute_pairwise_jaccard_measures(x_feat)
    
def select_reliable_negatives(x, y, k, t):

    k, t = int(k), float(t)

    global distances

    p = x[y == 1]
    u = x[y == 0]

    d_subset = np.concatenate([p, u]).T

    distances_subset = distances.copy()

    distances_subset[:, ~np.isin(np.arange(distances_subset.shape[1]), d_subset)] = np.nan

    topk = np.argsort(distances_subset, axis=1)[:, :k] # Indices of the k closest genes for each gene

    topk_is_unlabelled = np.isin(topk, u) 
    closest_unlabelled = topk_is_unlabelled[:, 0] # Condition 1: Closest gene is unlabelled
    topk_percent_unlabelled = np.mean(topk_is_unlabelled, axis=1)

    rn = np.where((closest_unlabelled & (topk_percent_unlabelled >= t)))[0]
    rn = np.intersect1d(rn, u)

    x = np.concatenate([p, rn])
    y = np.concatenate([np.ones(len(p)), np.zeros(len(rn))])

    idx = np.arange(len(y))
    np.random.shuffle(idx)
    x = x[idx]
    y = y[idx]

    return x, y


