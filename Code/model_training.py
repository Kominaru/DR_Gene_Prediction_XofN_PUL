import numpy as np
from sklearn.model_selection import StratifiedKFold
from imblearn.metrics import geometric_mean_score
from Code.data_processing import generate_features, resample_data

CV_INNER = 5


def cv_train_with_params(x_train, y_train, model, params, verbose=0, random_state=42):
    """
    Perform cross-validation training with specified parameters.

    Args:
        x_train (pandas.DataFrame): Training data features
        y_train (pandas.Series): Training data labels
        model: Model object with fit and predict methods
        params (dict): Dictionary of parameters
        verbose (int, optional): Verbosity mode
        random_state (int, optional): Random seed

    Returns:
        float: Mean GMean score of the cross-validation.

    """

    inner_skf = StratifiedKFold(
        n_splits=CV_INNER, shuffle=True, random_state=random_state
    )

    for _, (learn_idx, val_idx) in enumerate(inner_skf.split(x_train, y_train)):

        x_learn, x_val = x_train.iloc[learn_idx], x_train.iloc[val_idx]
        y_learn, y_val = y_train[learn_idx], y_train[val_idx]

        x_learn, y_learn = resample_data(
            x_learn,
            y_learn,
            method=params["sampling_method"],
            random_state=random_state,
        )

        x_learn, x_val = generate_features(
            x_learn, x_val, y_learn, y_val, params, seed=random_state
        )

        model.fit(x_learn, y_learn, verbose=0)
        pred_val = model.predict(x_val)

        score = geometric_mean_score(y_val, pred_val)

    return score.mean()
