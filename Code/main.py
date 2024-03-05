import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from Code.models import get_model
from Code.data_processing import load_data, generate_features, resample_data
from Code.hyperparameter_tuning import grid_search_hyperparams
from Code.metrics import compute_metrics

# Execution parameters
DATASET = "PathDip"
CLASSIFIER = "CAT"
CV_OUTER = 10
CV_INNER = 5
SEED = 42 if len(sys.argv) < 2 else int(sys.argv[1])
DO_GRID_SEARCH = True
HYPER_PARAMS = {
    "sampling_method": ["under"],
    "binary_threshold": [4],
    "use_original_features": [True],
    "use_xofn_features": [True],
    "xofn_min_sample_leaf": [2,5,10],
}


model = get_model(CLASSIFIER, SEED)

x, y = load_data(f"./Data/Datasets/{DATASET}.csv")

outer_cv = StratifiedKFold(n_splits=CV_OUTER, shuffle=True, random_state=SEED)

results = []

for k, (train_idx, test_idx) in enumerate(outer_cv.split(x, y)):

    print(f"OUTER FOLD {k}/{CV_OUTER}")
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    best_config_params = grid_search_hyperparams(HYPER_PARAMS, x_train, y_train, model)

    x_train, y_train = resample_data(
        x_train,
        y_train,
        method=best_config_params["sampling_method"],
        random_state=SEED,
    )

    x_train, x_test = generate_features(
        x_train, x_test, y_train, y_test, best_config_params, seed=SEED, verbose=1
    )

    model.fit(x_train, y_train, verbose=0)
    out_prob_test = model.predict_proba(x_test)[:, 1]

    metrics = compute_metrics(y_test, out_prob_test)

    print('\t', f"Sens: {metrics['Sensitivity']:.2}, Prec: {metrics['Precision']:.2}, Spec: {metrics['Specificity']:.2}, F1: {metrics['F1']:.2}, Gmean: {metrics['Gmean']:.2}, AUC: {metrics['AUC']:.2}")

    results.append(metrics)

for metric in results[0]:
    print(f"{metric}: {np.mean([result[metric] for result in results]):.3f}")
