import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from code.models import get_model, set_class_weights
from code.data_processing import load_data, generate_features, resample_data, store_data_features
from code.hyperparameter_tuning import grid_search_hyperparams
import code.pu_learning as pu_learning
import code.metrics as metrics


def run_experiment(PARAMS, random_state=42, neptune_run=None):

    random_state = int(random_state)

    x, y, gene_names = load_data(PARAMS["dataset"])

    pu_learning.compute_pairwise_jaccard_measures(x)
    x = store_data_features(x)

    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

    preds = []

    metrics_list = []

    for k, (train_idx, test_idx) in enumerate(outer_cv.split(x, y)):

        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        best_params = grid_search_hyperparams(PARAMS, x_train, y_train, random_state=random_state)

        print(f"Best params: {best_params}")

        # Log the best numeric hyperparameters
        if best_params["pu_learning"] == "similarity":
            neptune_run["hyperarameters/best_selected/pu_k"].append(best_params["pu_k"])
            neptune_run["hyperarameters/best_selected/pu_t"].append(best_params["pu_t"])
        elif best_params["pu_learning"] == "threshold":
            neptune_run["hyperarameters/best_selected/pu_t"].append(best_params["pu_t"])

        x_train, y_train = resample_data(
            x_train,
            y_train,
            method=best_params["sampling_method"],
            random_state=PARAMS["random_state"],
        )

        if best_params["pu_learning"]:
            if best_params["dataset"] == "GO":
                pu_learning.feature_selection_jaccard(
                    x_train,
                    y_train,
                    best_params["pul_num_features"],
                    best_params["pul_fs"],
                    classifier=best_params["classifier"],
                    sampling_method=best_params["sampling_method"],
                    random_state=random_state,
                )

            x_train, y_train = pu_learning.select_reliable_negatives(
                x_train, y_train, best_params["pu_learning"], best_params["pu_k"], best_params["pu_t"]
            )

        x_feat_train, x_feat_test = generate_features(
            x_train, x_test, y_train, y_test, best_params, random_state=random_state
        )

        model = get_model(PARAMS["classifier"], random_state)

        # If the model is a CatBoostClassifier, modify the class weights based on the number of positive and negative samples
        if PARAMS["classifier"] in ["CAT"]:
            model = set_class_weights(model, PARAMS["sampling_method"], y_train)
            model.fit(x_feat_train, y_train, verbose=0)
        elif PARAMS["classifier"] == "XGB":
            pos_weight = (len(y_train) - np.sum(y_train)) / np.sum(y_train)
            model.set_params(scale_pos_weight=pos_weight)
            model.fit(x_feat_train, y_train, verbose=0)
        else:
            model.fit(x_feat_train, y_train)

        out_prob_test = model.predict_proba(x_feat_test)[:, 1]

        preds += zip(test_idx, gene_names[test_idx], out_prob_test)

        fold_metrics = metrics.compute_metrics(y_test, out_prob_test)

        for metric, value in fold_metrics.items():
            neptune_run[f"metrics/run_{random_state}/fold_{k}/test/{metric}"] = value

        metrics_list.append(fold_metrics)

    for metric in metrics_list[0].keys():
        neptune_run[f"metrics/run_{random_state}/global/test/" + metric] = np.mean(
            [fold[metric] for fold in metrics_list]
        )

    preds = pd.DataFrame(preds, columns=["id", "gene", "prob"])

    # Save the predictions to neptune
    preds_sorted = preds.sort_values(by="prob", ascending=False)
    preds_sorted.to_csv("preds.csv", index=False)
    neptune_run[f"predictions/run_{random_state}"].upload("preds.csv")

    preds = preds.sort_values(by="id")

    return metrics_list, preds
