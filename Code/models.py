# Define the model and its grid parameters
from typing import Union
from imblearn.ensemble import EasyEnsembleClassifier, BalancedRandomForestClassifier
from catboost import CatBoostClassifier
import numpy as np
from xgboost import XGBClassifier


def get_model(model: str, random_state: int) -> Union[
    EasyEnsembleClassifier,
    BalancedRandomForestClassifier,
    CatBoostClassifier,
    XGBClassifier,
]:
    """
    Returns a classifier model based on the given CLASSIFIER parameter.

    Parameters:
        CLASSIFIER (str): The type of classifier to be returned.
        random_state (int): The random seed for reproducibility.

    Returns:
        model (Union[EasyEnsembleClassifier, BalancedRandomForestClassifier, CatBoostClassifier, XGBClassifier]): The classifier model.

    Raises:
        ValueError: If the CLASSIFIER parameter is not one of 'EEC', 'BRF', 'CAT', or 'XGB'.
    """
    if model == "EEC":
        model = EasyEnsembleClassifier(random_state=random_state)
    elif model == "BRF":
        model = BalancedRandomForestClassifier(
            random_state=random_state,
            n_jobs=4,
            sampling_strategy=1.0,
            replacement=True,
            n_estimators=500,
            bootstrap=True
        )
    elif model == "CAT":
        model = CatBoostClassifier(random_state=random_state, n_estimators=500)
    elif model == "XGB":
        model = XGBClassifier(random_state=random_state)
    else:
        raise ValueError(
            f"Invalid classifier type {model}. Please choose one of 'EEC', 'BRF', 'CAT', or 'XGB'."
        )

    return model


def set_class_weights(model, weight : Union[str, float] = "balanced", y_train: np.ndarray = None):
    """
    Set the class weights for the model

    Parameters:
        model: The classifier model.
        weight (Union[str, float]): The weighting strategy.
            - If "balanced", y_train is required and the class weights are set as:
                - w_pos = len(y_train) / (2 * sum(y_train))
                - w_neg = len(y_train) / (2 * (len(y_train) - sum(y_train)))
            - If a float, the class weights are set as:
                - w_pos = weight
                - w_neg = 1

        y_train: The training labels. Used only if weight is "balanced".

    Returns:
        - The model with the class weights set

    """

    try:
        weight = float(weight)
        model.set_params(class_weights=[1, float(weight)])
    except ValueError:
        pass

    if weight == "balanced":

        w_pos = len(y_train) / (2 * np.sum(y_train))
        w_neg = len(y_train) / (2 * (len(y_train) - np.sum(y_train)))

        model.set_params(class_weights=[w_neg, w_pos])

    return model