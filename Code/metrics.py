from sklearn.metrics import auc, f1_score, precision_recall_curve, precision_score, roc_auc_score
import pandas as pd
from imblearn.metrics import geometric_mean_score, sensitivity_score, specificity_score

def compute_metrics(y_true, y_prob):
    """
    Computes the metrics for the given true labels and predicted probabilities.

    Parameters:
    - y_true: array-like of shape (n_samples,)
        True labels.
    - y_prob: array-like of shape (n_samples,)
        Predicted probabilities.
    
    Returns:
    - metrics: dict
    
    """
    y_pred = (y_prob >= 0.5).astype(int)  # Threshold probabilities to obtain binary predictions

    test_sens = sensitivity_score(y_true, y_pred)
    test_spec = specificity_score(y_true, y_pred)
    test_gmean = geometric_mean_score(y_true, y_pred)
    test_auc_roc = roc_auc_score(y_true, y_prob)
    test_precision = precision_score(y_true, y_pred)
    test_f1 = f1_score(y_true, y_pred)
    
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    test_auc_pr = auc(recalls, precisions)

    metrics = {
        "sensitivity": test_sens,
        "specificity": test_spec,
        "precision": test_precision,
        "f1": test_f1,
        "gmean": test_gmean,
        "auc_roc": test_auc_roc,
        "auc_pr": test_auc_pr
    } 

    return metrics