from sklearn.metrics import f1_score, precision_score, roc_auc_score
import pandas as pd
from imblearn.metrics import geometric_mean_score, sensitivity_score, specificity_score

def compute_metrics(y_true, y_prob):
    """
    Compute metrics for the given true labels and predicted probabilities.

    Parameters:
    - y_true: array-like of shape (n_samples,)
        True labels.
    - y_prob: array-like of shape (n_samples,)
        Predicted probabilities.

    Returns:
    - metrics: dict
        Dictionary containing the computed metrics.
    """
    y_pred = (y_prob >= 0.5).astype(int)  # Threshold probabilities to obtain binary predictions

    test_sens = sensitivity_score(y_true, y_pred)
    test_spec = specificity_score(y_true, y_pred)
    test_gmean = geometric_mean_score(y_true, y_pred)
    test_auc = roc_auc_score(y_true, y_prob)
    test_precision = precision_score(y_true, y_pred)
    test_f1 = f1_score(y_true, y_pred)

    return {
        "Sensitivity": test_sens,
        "Specificity": test_spec,
        "Precision": test_precision,
        "F1": test_f1,
        "Gmean": test_gmean,
        "AUC": test_auc
    }