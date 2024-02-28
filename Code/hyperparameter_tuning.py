import itertools
from Code.model_training import cv_train_with_params


def grid_search_hyperparams(HYPER_PARAMS, x_train, y_train, model, seed=42):
    """
    Perform grid search to find the best hyperparameters for a given model.

    Parameters:
        - HYPER_PARAMS (dict): A dictionary containing the hyperparameters to be tuned.
        - x_train (array-like): The input features for training.
        - y_train (array-like): The target variable for training.
        - model: Model object with fit and predict methods.
        - cv_inner (int, optional): The number of folds for inner cross-validation.
        - seed (int, optional): The random seed for reproducibility. Default is 42.

    Returns:
    dict: The best hyperparameters found during grid search.
    """
    best_config = {"params": None, "score": 0}

    print('\t',f"Testing {len(list(itertools.product(*HYPER_PARAMS.values())))} configurations...")

    for params in itertools.product(*HYPER_PARAMS.values()):
        params = {k: v for k, v in zip(HYPER_PARAMS, params)}

        score = cv_train_with_params(
            x_train, y_train, model, params, verbose=0, random_state=seed
        )

        print('\t\t',f"Score: {score}, Params: {params}")

        if score > best_config["score"]:
            best_config = {"params": params, "score": score}

    return best_config["params"]
