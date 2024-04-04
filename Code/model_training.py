import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
import code.pu_learning as pu_learning
from code.data_processing import generate_features, resample_data
from code.models import get_model, set_class_weights
from sklearn.metrics import f1_score
from imblearn.metrics import geometric_mean_score

CV_INNER = 5


def cv_train_with_params(x_train, y_train, classifier, params, random_state=42, verbose=0):
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

    inner_skf = StratifiedKFold(n_splits=CV_INNER, shuffle=True, random_state=random_state)

    neptune_run = params["neptune_run"]

    score = []

    for _, (learn_idx, val_idx) in enumerate(inner_skf.split(x_train, y_train)):

        t = time.time()

        x_learn, x_val = x_train[learn_idx], x_train[val_idx]
        y_learn, y_val = y_train[learn_idx], y_train[val_idx]

        x_learn, y_learn = resample_data(
            x_learn,
            y_learn,
            method=params["sampling_method"],
            random_state=params["random_state"],
        )

        if params["pu_learning"]:

            t1 = time.time()
            if params["dataset"] == "GO":
                pu_learning.feature_selection_jaccard(
                    x_learn,
                    y_learn,
                    params["pul_num_features"],
                    params["pul_fs"],
                    classifier=classifier,
                    sampling_method=params["sampling_method"],
                    random_state=random_state,
                )

            neptune_run["metrics/fs_time_overhead"].append(time.time() - t1)

            x_learn, y_learn = pu_learning.select_reliable_negatives(
                x_learn, y_learn, params["pu_learning"], params["pu_k"], params["pu_t"]
            )

            # Log the number of reliable negatives
            neptune_run["metrics/innercv_num_reliable_negatives"].append(np.sum(y_learn == 0) / len(y_learn))

            # If not enough reliable negatives are found, return a score of 0
            if np.sum(y_learn == 0) == 0:
                return 0

        x_learn_feat, x_val_feat = generate_features(x_learn, x_val, y_learn, y_val, params, random_state=random_state)

        model = get_model(classifier, random_state=random_state)

        if classifier == "CAT":
            model = set_class_weights(model, params["sampling_method"], y_learn)
            model.fit(x_learn_feat, y_learn, verbose=0)
        elif classifier == "XGB":
            pos_weight = (len(y_learn) - np.sum(y_learn)) / np.sum(y_learn)
            model.set_params(scale_pos_weight=pos_weight)
            model.fit(x_learn_feat, y_learn, verbose=0)
        else:
            model.fit(x_learn_feat, y_learn)

        pred_val = model.predict(x_val_feat)

        if params["pu_learning"]:
            score.append(f1_score(y_val, pred_val))
        else:
            score.append(geometric_mean_score(y_val, pred_val))

        neptune_run["metrics/inner_fold_time"].append(time.time() - t)

    return np.mean(score)
