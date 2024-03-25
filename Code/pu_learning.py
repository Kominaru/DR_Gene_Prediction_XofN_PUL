
import numpy as np
import sklearn
from code.data_processing import get_data_features
from sklearn.metrics import pairwise_distances
from code.models import get_model, set_class_weights
from skrebate import ReliefF
from sklearn.ensemble import RandomForestClassifier

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
    
def select_reliable_negatives(x, y, method, k, t, random_state=42):

    if method == 'similarity':
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
    
    elif method == "threshold":
        
        p = x[y == 1]
        u = x[y == 0]

        # Split u into k subsets
        u_split = np.array_split(u, k)

        rn = []

        for i in range(k):

            # Train a random forest on the positive samples and the i-th subset of unlabelled samples
            x_i = np.concatenate([p, u_split[i]])
            y_i = np.concatenate([np.ones(len(p)), np.zeros(len(u_split[i]))])

            x_i_feat = get_data_features(x_i)

            model = RandomForestClassifier(random_state=random_state)
            model.fit(x_i_feat, y_i)

            u_i_feat = get_data_features(u_split[i])

            model_predictions = model.predict_proba(u_i_feat)[:, 1]

            # Print the average probability of the positive class
            # print(np.mean(model.predict_proba(x_i_feat)[np.where(y_i == 1), 1]))
            # # Print the average probability of the negative class
            # print(np.mean(model.predict_proba(x_i_feat)[np.where(y_i == 0), 1]))

            rn.append(u_split[i][model_predictions <= t])

        rn = np.concatenate(rn)

        x = np.concatenate([p, rn])
        y = np.concatenate([np.ones(len(p)), np.zeros(len(rn))])

        idx = np.arange(len(y))
        np.random.shuffle(idx)
        x = x[idx]
        y = y[idx]

        return x, y

        
            



