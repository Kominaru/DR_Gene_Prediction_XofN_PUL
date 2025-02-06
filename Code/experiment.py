import numpy as np
import pandas as pd
from typing import Literal, Union
from sklearn.model_selection import StratifiedKFold
import neptune
from code.model_training import train_a_model
from code.data_processing import load_data, store_data_features
from code.hyperparameter_tuning import grid_search_hyperparams
from code.neptune_utils import upload_preds_to_neptune
from codecarbon.emissions_tracker import EmissionsTracker
import code.metrics as metrics
import code.pu_learning as pul


def run_experiment(
    dataset: str = None,
    classifier: str = None,
    pul_num_features: int = 0,
    pu_learning: Union[Literal["similarity", "threshold"], bool] = False,
    search_space: dict = None,
    random_state: int = 42,
    neptune_run: Union[neptune.Run, None] = None,
    tracker: EmissionsTracker = None,
):
    """
    Run a single experiment with the given parameters performing a nested 10x5 cross-validation.

    Parameters
    ----------
    - dataset : str
      - Name of the dataset to be used (default: None)
    - classifier : str
      - Acronym of the classifier to use. Can be one of 'EEC', 'BRF', 'CAT', or 'XGB' (default: None)
    - pul_num_features : int
      - Number of features to be selected to select reliable negatives with PU Learning
        - If 0, all features will be used.
    - pu_learning : str | bool (default: False)
        - The PU learning method to be used.
        - If False, no PU learning will be performed.
        - If str, the PU learning method to be used. Can be one of 'similarity' or 'threshold'.
    - search_space : dict
        - The search space for the hyperparameters to be tuned.
    - random_state : int
    - neptune_run : neptune.Run | None
        - If passed, the Neptune run to log the experiment to (default: None)

    Returns
    -------
    - metrics : dict
      - Metrics computed for each fold of the experiment.
    - preds : pd.DataFrame
      - Predictions for each gene in the dataset.
    """

    random_state = int(random_state)

    x, y, gene_names = load_data(dataset)

    if pu_learning and random_state == 14:
        pul.compute_pairwise_jaccard_measures(x)

    x = store_data_features(x)

    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

    experiment_preds, experiment_metrics = [], []

    for k, (train_idx, test_idx) in enumerate(outer_cv.split(x, y)):

        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        tracker.stop()

        best_params = grid_search_hyperparams(
            x_train,
            y_train,
            classifier=classifier,
            random_state=random_state,
            pu_learning=pu_learning,
            pul_num_features=pul_num_features,
            search_space=search_space,
            neptune_run=neptune_run,
        )

        tracker.start()

        pred_test = train_a_model(
            x_train,
            y_train,
            x_test,
            classifier,
            random_state=random_state,
            pu_learning=pu_learning,
            pul_num_features=pul_num_features,
            pul_k=best_params["pu_k"],
            pul_t=best_params["pu_t"],
        )

        experiment_preds += zip(test_idx, gene_names[test_idx], pred_test)
        fold_metrics = metrics.log_metrics(y_test, pred_test, neptune_run=neptune_run, run_number=random_state, fold=k)
        experiment_metrics.append(fold_metrics)

    experiment_preds = pd.DataFrame(experiment_preds, columns=["id", "gene", "prob"])

    # Save the predictions to neptune
    if neptune_run:
        upload_preds_to_neptune(
            preds=experiment_preds,
            random_state=random_state,
            neptune_run=neptune_run,
        )

    # Log the metrics to neptune
    for metric in experiment_metrics[0].keys():
        if neptune_run:
            neptune_run[f"metrics/run_{random_state}/global/test/{metric}"] = np.mean(
                [fold_metrics[metric] for fold_metrics in experiment_metrics]
            )
        else:
            print(
                f"metrics/run_{random_state}/global/test/{metric}: {np.mean([fold_metrics[metric] for fold_metrics in experiment_metrics])}"
            )

    return experiment_metrics, experiment_preds
