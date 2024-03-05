import numpy as np
import pandas as pd
from sklearn import preprocessing
from Code.X_of_N_features import construct_xofn_features, compute_xofn
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import make_scorer
from imblearn.metrics import geometric_mean_score


def load_data(dataset_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from a CSV file.

    Parameters:
    dataset_path (str): The path to the CSV file.

    Returns:
    tuple: A tuple containing the features (x) and the labels (y).
    """
    raw_df = pd.read_csv(dataset_path)
    x, y = raw_df.iloc[:, 1:-1], raw_df.iloc[:, -1]
    y = np.logical_not(preprocessing.LabelEncoder().fit(y).transform(y)).astype(int)
    return x, y


def generate_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test,
    params: dict,
    seed: int = 42,
    verbose: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate features for training and testing data based on specified parameters.

    Parameters:
        - x_train (pd.DataFrame): Training data features.
        - x_test (pd.DataFrame): Testing data features.
        - y_train (pd.Series): Training data labels.
        - y_test (pd.Series): Testing data labels.
        - params (dict): Dictionary containing parameters for feature generation.
        - seed (int, optional): Random seed.
        - verbose (int, optional): Whether to print number and type of features used.
    Returns:
        - x_train_temp (pd.DataFrame): Transformed training data features.
        - x_test_temp (pd.DataFrame): Transformed testing data features.
    """
    x_train_temp = pd.DataFrame()
    x_test_temp = pd.DataFrame()

    x_train = x_train.loc[:, x_train.sum() >= params["binary_threshold"]]
    x_test = x_test.loc[:, x_train.columns]

    if verbose: print('\t\t\t',f"Using features:", end="")

    if params["use_original_features"]:
        x_train_temp = pd.concat([x_train_temp, x_train], axis=1)
        x_test_temp = pd.concat([x_test_temp, x_test], axis=1)

        if verbose: print(f"+ {x_train.shape[1]} OG", end="")

    if params["use_xofn_features"]:
        xofn_features = construct_xofn_features(
            x_train, y_train, min_samples_leaf=params["xofn_min_sample_leaf"], seed=seed
        )

        if params["use_original_features"]:
            xofn_features = [path for path in xofn_features if len(path) > 1]

        x_train_xofn = compute_xofn(x_train, xofn_features)
        x_test_xofn = compute_xofn(x_test, xofn_features)

        x_train_temp = pd.concat([x_train_temp, x_train_xofn], axis=1)
        x_test_temp = pd.concat([x_test_temp, x_test_xofn], axis=1)

        if verbose: print(f"+ {x_train_xofn.shape[1]} X-of-N", end="")

    if verbose: print()

    return x_train_temp, x_test_temp


def resample_data(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    method: str = "normal",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Resample the data based on the specified method.

    Parameters:
        - x_train (pd.DataFrame): Training data features.
        - y_train (pd.DataFrame): Training data labels.
        - method (str, optional): Resampling method (one of "normal", "weighted", "under")
            - "normal", "weighted": No resampling
            - "under": Random under-sampling
        - random_state (int, optional): Random seed

    Returns:
        - x_train (pd.DataFrame): Resampled training data features.
        - y_train (pd.DataFrame): Resampled training data labels.
    """

    if method in ["normal", "weighted"]:  # NORMAL
        pass

    elif method == "under":  # RANDOM UNDER SAMPLING
        rus = RandomUnderSampler(random_state=random_state)
        x_train, y_train = rus.fit_resample(x_train, y_train)

    return x_train, y_train


geometric_mean_scorer = make_scorer(geometric_mean_score, greater_is_better=True)
