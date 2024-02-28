# Define the model and its grid parameters
from typing import Union
from imblearn.ensemble import EasyEnsembleClassifier, BalancedRandomForestClassifier
from catboost import CatBoostClassifier
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
        model = BalancedRandomForestClassifier(random_state=random_state, n_jobs=4)
    elif model == "CAT":
        model = CatBoostClassifier(
            random_state=random_state, n_estimators=500, auto_class_weights="Balanced"
        )
    elif model == "XGB":
        model = XGBClassifier(random_state=random_state)
    else:
        raise ValueError(
            "Invalid classifier type. Please choose one of 'EEC', 'BRF', 'CAT', or 'XGB'."
        )

    return model
