import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn import preprocessing
from catboost import CatBoostClassifier
from imblearn.metrics import geometric_mean_score, sensitivity_score, specificity_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from sklearn.metrics import f1_score, precision_score, roc_auc_score

from Code.ML_Procedures.auxilitary_functions import resample_data, geometric_mean_scorer
from Code.ML_Procedures.X_of_N_features import construct_xofn_features, compute_xofn

# Execution parameters

DATASET = 'GoDataset' # 'PathDip', 'GoDataset'
CLASSIFIER = 'CAT' 
SAMPLING_METHOD = 'weighted'

CV_OUTER = 10
CV_INNER = 5

SEED = 42

USE_FEATURES = {
    "Original": True,
    "X-of-N": True
}

FILTER_MIN_OCC = True
MIN_OCCS = [3,4,5]

# Define the model and its grid parameters

if CLASSIFIER == 'EEC':
    model = EasyEnsembleClassifier(random_state=SEED)
    model_grid_params = {}
elif CLASSIFIER == 'BRF':
    model = BalancedRandomForestClassifier(random_state=SEED, n_jobs=4)
    model_grid_params = {}
elif CLASSIFIER == 'CAT':
    model = CatBoostClassifier(random_state=SEED, n_estimators=500, auto_class_weights='Balanced')
    model_grid_params = {'n_estimators': [500]}
elif CLASSIFIER == 'XGB':
    model = XGBClassifier(random_state=SEED)
    model_grid_params = {}

# Prepare dataset
    
raw_df = pd.read_csv(f'./Data/Datasets/{DATASET}.csv')
x, y = raw_df.iloc[:,1:-1], raw_df.iloc[:,-1]

y = np.logical_not(preprocessing.LabelEncoder().fit(y).transform(y)).astype(int)

print(f"{DATASET}: {x.shape[1]} features, {x.shape[0]} genes. {sum(y)}/{len(y)-sum(y)} P/N. NA: {x.isna().sum().sum()}, Inf: {np.isinf(x).sum().sum()}")

outer_cv = StratifiedKFold(n_splits=CV_OUTER, shuffle=True, random_state=SEED)

results = []

for k, (train_idx, test_idx) in enumerate(outer_cv.split(x, y)):

    print(f"OUTER FOLD {k}/{CV_OUTER}")
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    x_train, y_train = resample_data(x_train, y_train, method = SAMPLING_METHOD, seed=SEED) 

    best_gmean = 0

    if FILTER_MIN_OCC and len(MIN_OCCS) > 1:
        print('\t',"Starting hyperparam search...")
        for min_occ in MIN_OCCS:
            print('\t\t',f"T={min_occ} |", end="")

            # Drop features with less than min_occ occurrences
            # One feature per column, all features are binary
            x_train_in = x_train.loc[:, x_train.sum() >= min_occ]
            x_test_in = x_test.loc[:, x_train_in.columns]
            print(f" {x_train_in.shape[1]} OG featuers ", end="")

            x_temp = pd.DataFrame()

            if USE_FEATURES["Original"]:
                x_train_temp = pd.concat([x_temp, x_train_in], axis=1)
                x_test_temp = pd.concat([x_temp, x_test_in], axis=1)
            
            if USE_FEATURES["X-of-N"]:
                xofn_features = construct_xofn_features(x_train_in, y_train, seed=SEED)

                if USE_FEATURES["Original"]:
                    # Remove X-of-N features that are already in the dataset (i.e. size 1)
                    xofn_features = [path for path in xofn_features if len(path) > 1]

                print(f"+ {len(xofn_features)} X-of-N features.")
                
                x_train_xofn = compute_xofn(x_train_in, xofn_features)
                x_test_xofn = compute_xofn(x_test_in, xofn_features)

                x_train_temp = pd.concat([x_train_temp, x_train_xofn], axis=1)
                x_test_temp = pd.concat([x_test_temp, x_test_xofn], axis=1)

            x_train_in = x_train_temp
            x_test_in = x_test_temp

            # Inner Cross Validation

            inner_skf = StratifiedKFold(n_splits=CV_INNER, shuffle=True, random_state = SEED)

            grid_search = GridSearchCV(estimator = model, param_grid = model_grid_params, scoring = geometric_mean_scorer, cv = inner_skf, verbose = 0, return_train_score = True)
            grid_search.fit(x_train_in, y_train, verbose = 0)

            if grid_search.best_score_ > best_gmean:
                best_gmean = grid_search.best_score_
                best_minoc = min_occ
                best_model = grid_search.best_estimator_
    else:
        best_model = model
        best_minoc = MIN_OCCS[-1]

    print('\t',"Training with best hyperparams...")

    x_train_temp = pd.DataFrame()
    x_test_temp = pd.DataFrame()

    if FILTER_MIN_OCC:
        x_train = x_train.loc[:, x_train.sum() >= best_minoc]
        x_test = x_test.loc[:, x_train.columns]

    if USE_FEATURES["Original"]:
        
        # Add columns to x_train_temp and x_test_temp
        x_train_temp = pd.concat([x_train_temp, x_train], axis=1)
        x_test_temp = pd.concat([x_test_temp, x_test],  axis=1)

        print(f"({x_train.shape[1]}) feat.", end="")


    if USE_FEATURES["X-of-N"]:
        xofn_features = construct_xofn_features(x_train, y_train, seed=SEED)

        if USE_FEATURES["Original"]:
            # If the original features are being used, discard X-of-N features that are already in the dataset (i.e. size 1)
            xofn_features = [path for path in xofn_features if len(path) > 1]

        x_train_xofn = compute_xofn(x_train, xofn_features)
        x_test_xofn = compute_xofn(x_test, xofn_features)

        x_train_temp = pd.concat([x_train_temp, x_train_xofn], axis=1)
        x_test_temp = pd.concat([x_test_temp, x_test_xofn], axis=1)

        print('\t\t', f" + {x_train_xofn.shape[1]} X-of-N features.")
    
    x_train = x_train_temp
    x_test = x_test_temp

    in_best_model = best_model.fit(x_train, y_train, verbose = 0)

    out_pred_test = in_best_model.predict(x_test)
    out_prob_test = in_best_model.predict_proba(x_test)[:,1]

    confusion_matrix = pd.crosstab(y_test, out_pred_test, rownames=['Actual'], colnames=['Predicted'])

    test_sens = sensitivity_score(y_test, out_pred_test)
    test_spec = specificity_score(y_test, out_pred_test)
    test_gmean = geometric_mean_score(y_test, out_pred_test)
    test_auc = roc_auc_score(y_test, out_prob_test)
    test_precision = precision_score(y_test, out_pred_test)
    test_f1 = f1_score(y_test, out_pred_test)

    print('\t',f"Fold test Sens: {test_sens} | Spec: {test_spec} | Gmean: {test_gmean} | AUC: {test_auc}")

    metrics = {
        "Test": {
            "ConfusionMatrix": confusion_matrix,
            "Sensitivity": test_sens,
            "Specificity": test_spec,
            "Precision": test_precision,
            "F1": test_f1,
            "Gmean": test_gmean,
            "AUC": test_auc
        }
    }
    
    results.append({
        "Metrics": metrics,
    })

test_sens = np.average([results[i]["Metrics"]["Test"]["Sensitivity"] for i in range(CV_OUTER)])
test_spec = np.average([results[i]["Metrics"]["Test"]["Specificity"] for i in range(CV_OUTER)])
test_precision = np.average([results[i]["Metrics"]["Test"]["Precision"] for i in range(CV_OUTER)])
test_f1 = np.average([results[i]["Metrics"]["Test"]["F1"] for i in range(CV_OUTER)])
test_gmean = np.average([results[i]["Metrics"]["Test"]["Gmean"] for i in range(CV_OUTER)])
test_auc = np.average([results[i]["Metrics"]["Test"]["AUC"] for i in range(CV_OUTER)])

confusion_matrix = sum([results[i]["Metrics"]["Test"]["ConfusionMatrix"] for i in range(CV_OUTER)])

print("="*50)

print(f"Test Sensitivity: {test_sens:.3}")
print(f"Test Precision: {test_precision:.3}")
print(f"Test Specificity: {test_spec:.3}")
print(f"Test F1: {test_f1:.3}")
print(f"Test Gmean: {test_gmean:.3}")
print(f"Test AUC: {test_auc:.3}")