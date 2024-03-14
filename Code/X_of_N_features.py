import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from sklearn.feature_selection import mutual_info_classif


def tree_to_xofn(tree):
    """
    Convert a decision tree to a set of paths (attribute-value pairs) that can be used to create X-of-N features

    Parameters:
        tree (sklearn.tree._tree.Tree): Decision tree

    Returns:
        set: Set of sets, where each set is a path (attribute-value pairs) in the decision tree to be used as X-of-N features
    """

    all_paths = set()

    def tree_to_xofn_aux(node, curr_path: set):
        if tree.feature[node] == _tree.TREE_UNDEFINED:
            curr_path = frozenset(curr_path)
            all_paths.add(curr_path)
        else:
            feature = tree.feature[node]

            left_path = curr_path.copy()
            right_path = curr_path.copy()

            left_path.add((feature,0))
            right_path.add((feature, 1))

            tree_to_xofn_aux(tree.children_left[node], left_path)
            tree_to_xofn_aux(tree.children_right[node], right_path)

    tree_to_xofn_aux(0, set())

    return all_paths


def create_xofn(x, path):
    """
    Create the X_of_N feature for a given path (set of attribute-value pairs)

    Parameters:
        path (set): Set of attribute-value pairs (int, int)
        x (np.array): Dataset features

    Returns:
        np.array (n,1): X-of-N feature (n is the number of samples in the dataset)
    """
    x_of_n = np.zeros(x.shape[0])
    for avv in path:

        if avv[1]:
            x_of_n += x[:, avv[0]]
        else:
            x_of_n += 1 - x[:, avv[0]]

    return np.expand_dims(x_of_n, axis=1)


def prune_xofn(x, y, path):
    """
    Prune the X-of-N feature for a given path (set of attribute-value pairs) through IG maximization

    Parameters:
        x (np.array): Dataset features
        y (np.array): Dataset labels
        path (set): Set of attribute-value pairs (int, int)

    Returns:
        set: Pruned X-of-N feature (set of attribute-value pairs)
    """
    path_best = path.copy()
    path_prev = path.copy().union({(-1, -1)})

    # Compute the X-of-N feature for the given path
    xofn_best = create_xofn(x, path_best)

    # Stop prunning when the X-of-N feature couldn't be reduced or it has only one feature
    while len(path_best) > 2 and len(path_best) < len(path_prev):

        path_size_best = path_best.copy()
        xofn_size_best = xofn_best
        path_prev = path_best.copy()

        for av in path_best:  # Try to remove each feature from the X-of-N feature

            path_temp = path_best.copy() - {av}

            if av[1]:
                xofn_temp = xofn_best - x[:, av[0]].reshape(-1, 1)
            else:
                xofn_temp = xofn_best - 1 + x[:, av[0]].reshape(-1, 1)

            # If the IG of the reduced X-of-N feature is higher than the best reduced X-of-N feature of this size,
            # update the best reduced X-of-N feature
            ig_temp = mutual_info_classif(xofn_temp, y, discrete_features=True)[0]
            ig_size_best = mutual_info_classif(
                xofn_size_best, y, discrete_features=True
            )[0]

            if ig_temp >= ig_size_best:
                path_size_best = path_temp.copy()
                xofn_size_best = xofn_temp

        path_best = path_size_best.copy()
        xofn_best = xofn_size_best

    return path_best


def construct_xofn_features(x, y, min_samples_leaf, seed):
    """
    Construct X-of-N features from the given dataset. This has the following steps:
        1) Train a decision tree with class samples weighted by the class imbalance
        2) Generate a X-of-N feature for each path in the decision tree ({AV1, AV2, ..., AVn})
        3) Prune attribute-value pairs from each X-of-N feature through IG maximization

    Parameters:
        x (np.array): Dataset features
        y (np.array): Dataset labels
        min_samples_leaf (int): Minimum leaf coverage for the initial decision tree
        seed (int): Random seed

    Returns:
        set: Set of pruned X-of-N features. Each X-of-N feature is a set of attribute-value pairs (int, int)
    """

    # 1)Train decision tree with weighted samples
    clf = DecisionTreeClassifier(
        min_samples_leaf=min_samples_leaf, class_weight="balanced", random_state=seed
    )
    clf.fit(x, y)

    # Make sure the features are an array
    x = np.array(x)

    # 2) Generate X-of-N features for each path in the decision tree

    all_paths = tree_to_xofn(clf.tree_)

    # 3) Prune attribute-value pairs from each X-of-N feature through IG maximization
    path_final = set()
    for path in all_paths:  # Prune each X-of-N feature
        path_pruned = prune_xofn(x, y, path)
        path_final.add(path_pruned)

    return path_final


def compute_xofn(x, path_final):
    """
    Compute the requested X-of-N features for the given dataset

    Parameters:
        x (np.array): Dataset features
        path_final (set): Set of pruned X-of-N features

    Returns:
        pd.DataFrame: Dataset with the requested X-of-N features computed from the original dataset according to the pruned X-of-N features
    """
    # Create dataset with X-of-N features
    x_arr = np.array(x)

    x_of_n = np.concatenate([create_xofn(x_arr, path) for path in path_final], axis=1)
    x_of_n = pd.DataFrame(
        x_of_n, columns=[f"X_of_N_{i+1}" for i in range(x_of_n.shape[1])]
    )

    if type(x) == pd.DataFrame:
        x_of_n.index = x.index

    return x_of_n
