import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from Code.models import get_model
from Code.data_processing import load_data, generate_features, resample_data
from Code.hyperparameter_tuning import grid_search_hyperparams
from Code.metrics import compute_metrics
import argparse

# Execution parameters
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="Dataset name")
parser.add_argument("--classifier", type=str, help="Classifier name")
parser.add_argument("--random_state", type=int, help="Random seed")
args = parser.parse_args()

DATASET = args.dataset if args.dataset else "PathDIP"
CLASSIFIER = args.classifier if args.classifier else "CAT"
RANDOM_STATE = args.random_state if args.random_state else 42

CV_OUTER = 10
CV_INNER = 5
HYPER_PARAMS = {
    "sampling_method": ["normal"],
    "binary_threshold": [5],
    "use_original_features": [False],
    "use_xofn_features": [True],
    "xofn_min_sample_leaf": [5],
    "xofn_feature_type": ["numerical"],
    "max_xofn_size": [5],
}

print(f"""Running experiment with the following parameters""")
print(f"Dataset: {DATASET}")
print(f"Classifier: {CLASSIFIER}")
print(f"Random seed: {RANDOM_STATE}")
for param, value in HYPER_PARAMS.items(): print(f"{param}: {value}")



x, y = load_data(f"./Data/Datasets/{DATASET}.csv")

outer_cv = StratifiedKFold(n_splits=CV_OUTER, shuffle=True, random_state=RANDOM_STATE)

results = []

for k, (train_idx, test_idx) in enumerate(outer_cv.split(x, y)):

    print(f"OUTER FOLD {k}/{CV_OUTER}")
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    best_config_params = grid_search_hyperparams(HYPER_PARAMS, x_train, y_train, CLASSIFIER, RANDOM_STATE)

    x_train, y_train = resample_data(
        x_train,
        y_train,
        method=best_config_params["sampling_method"],
        random_state=RANDOM_STATE,
    )

    x_train, x_test = generate_features(
        x_train, x_test, y_train, y_test, best_config_params, random_state=RANDOM_STATE, verbose=1
    )

    model = get_model(CLASSIFIER, RANDOM_STATE)

    # If the model is a CatBoostClassifier, modify the class weights based on the number of positive and negative samples
    if CLASSIFIER == "CAT":
        w_pos = len(y_train) / (2 * np.sum(y_train))
        w_neg = len(y_train) / (2 * (len(y_train) - np.sum(y_train)))

        model.set_params(
            class_weights=[w_neg, w_pos])
    
    if CLASSIFIER == "CAT":
        if best_config_params["use_xofn_features"] and best_config_params["xofn_feature_type"] == "categorical":
            x_of_n_features = [col for col in x_train.columns if "X_of_N" in col]
            model.fit(x_train, y_train, verbose=0, cat_features=x_of_n_features)
        else:
            model.fit(x_train, y_train, verbose=0)
    else:
        model.fit(x_train, y_train)

    ###################
    # TEMPORARY TESTING
    
    top_feature_indices = np.argsort(model.feature_importances_)[::-1][:50]

    # Get the names of the features
    feature_names = x_train.columns

    # Log the most important features and their types (original or X-of-N)
    with open("feature_importances.txt", "a") as f:
        f.write("="*20 + "\n")
        f.write(f"Fold {k}\n")
        for idx in top_feature_indices:
            feature_name = feature_names[idx]
            feature_type = "Original" if feature_name.startswith("X-of-N") else "X-of-N"
            feature_importance = model.feature_importances_[idx]
            f.write(f"{feature_name} ({feature_type}): {feature_importance}\n")
        
    out_prob_test = model.predict_proba(x_test)[:, 1]

    metrics = compute_metrics(y_test, out_prob_test)

    print('\t', f"Sens: {metrics['Sensitivity']:.2}, Prec: {metrics['Precision']:.2}, Spec: {metrics['Specificity']:.2}, F1: {metrics['F1']:.2}, Gmean: {metrics['Gmean']:.2}, AUC: {metrics['AUC']:.2}")

    results.append(metrics)

for metric in results[0]:
    print(f"{metric}: {np.mean([result[metric] for result in results]):.3f}")
