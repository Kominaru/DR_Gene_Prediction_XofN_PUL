import time
from typing import Literal, Union
import neptune
import numpy as np
from sklearn.model_selection import StratifiedKFold
from code.data_processing import generate_features
from code.models import get_model, set_class_weights
from sklearn.metrics import f1_score
from imblearn.metrics import geometric_mean_score
import code.pu_learning as pul

CV_INNER = 5


def cv_train_with_params(
    x_train,
    y_train,
    classifier,
    random_state=42,
    pu_learning=False,
    pul_num_features=None,
    pul_k=None,
    pul_t=None,
):
    """
    Perform cross-validation training with specified parameters.

    Args:
        x_train (pandas.DataFrame): Training data features
        y_train (pandas.Series): Training data labels
        classifier: String representing the model to be used.
        params (dict): Dictionary of parameters
        verbose (int, optional): Verbosity mode
        random_state (int, optional): Random seed

    Returns:
        float: Mean AUC score of the cross-validation.

    """

    inner_skf = StratifiedKFold(
        n_splits=CV_INNER, shuffle=True, random_state=random_state
    )

    score = []

    for _, (learn_idx, val_idx) in enumerate(inner_skf.split(x_train, y_train)):

        x_learn, x_val = x_train[learn_idx], x_train[val_idx]
        y_learn, y_val = y_train[learn_idx], y_train[val_idx]

        pred_val = train_a_model(
            x_learn,
            y_learn,
            x_val,
            classifier,
            random_state=random_state,
            pu_learning=pu_learning,
            pul_num_features=pul_num_features,
            pul_k=pul_k,
            pul_t=pul_t,
        )

        if pu_learning:
            score.append(f1_score(y_val, pred_val > 0.5))
        else:
            score.append(geometric_mean_score(y_val, pred_val > 0.5))

    return np.mean(score)


def train_a_model(
    x_train,
    y_train,
    x_test,
    classifier: Literal["CAT", "BRF", "XGB", "EEC"],
    random_state=42,
    pu_learning=False,
    pul_num_features=None,
    pul_k: int = None,
    pul_t: float = None,
):
    """
    Train a model with the specified parameters.

    Args:
        x_train (pandas.DataFrame): Training data features
        y_train (pandas.Series): Training data labels
        x_test (pandas.DataFrame): Test data features
        classifier: String representing the model to be used.
        params (dict): Dictionary of parameters
        verbose (int, optional): Verbosity mode
        random_state (int, optional): Random seed

    Returns:
        model: The trained model

    """

    if pu_learning:
        pul.feature_selection_jaccard(
            x_train,
            y_train,
            pul_num_features,
            classifier=classifier,
            random_state=random_state,
        )

        x_train, y_train = pul.select_reliable_negatives(
            x_train,
            y_train,
            pu_learning,
            pul_k,
            pul_t,
        )

    x_train, x_test = generate_features(x_train, x_test)

    model = get_model(classifier, random_state=random_state)

    if classifier == "CAT":
        model = set_class_weights(model, y_train)
        model.fit(x_train, y_train, verbose=0)
    elif classifier == "XGB":
        pos_weight = (len(y_train) - np.sum(y_train)) / np.sum(y_train)
        model.set_params(scale_pos_weight=pos_weight)
        model.fit(x_train, y_train, verbose=0)
    else:
        model.fit(x_train, y_train)

    probs = model.predict_proba(x_test)[:, 1]

    return probs
