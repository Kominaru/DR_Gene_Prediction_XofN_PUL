import numpy as np
from sklearn.model_selection import StratifiedKFold
from Code.data_processing import generate_features, resample_data
from Code.models import get_model
from sklearn.metrics import roc_auc_score

CV_INNER = 5


def cv_train_with_params(x_train, y_train, classifier, params, verbose=0, random_state=42):
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

    auc = []

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

        model = get_model(classifier, random_state=random_state)

        if classifier == "CAT":
            w_pos = len(y_learn) / (2 * np.sum(y_learn))
            w_neg = len(y_learn) / (2 * (len(y_learn) - np.sum(y_learn)))

            model.set_params(
                class_weights=[w_neg, w_pos]
            )
        if classifier == "CAT":
            
            #If X-of-N features are used and categorical is activated,
            #indicate that the features are categorical

            if params["use_xofn_features"] and params["xofn_feature_type"] == "categorical":
                x_of_n_features = [col for col in x_learn.columns if "X_of_N" in col]
                print(x_of_n_features)
                model.fit(x_learn, y_learn, verbose=0, cat_features=x_of_n_features)
            else:
                model.fit(x_learn, y_learn, verbose=0)
        else:
            model.fit(x_learn, y_learn)
            
        pred_val = model.predict(x_val)

        auc.append(roc_auc_score(y_val, pred_val))

    return np.mean(auc)
