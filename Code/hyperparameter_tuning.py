import itertools
from Code.model_training import cv_train_with_params


def grid_search_hyperparams(HYPER_PARAMS, x_train, y_train, model, seed):
    """
    Perform grid search to find the best hyperparameters for a given model.

    Parameters:
        - HYPER_PARAMS (dict): A dictionary containing the hyperparameters to be tuned.
        - x_train (array-like): The input features for training.
        - y_train (array-like): The target variable for training.
        - model: String representing the model to be used.
        - cv_inner (int, optional): The number of folds for inner cross-validation.
        - seed (int, optional): The random seed for reproducibility. Default is 42.

    Returns:
    dict: The best hyperparameters found during grid search.
    """
    best_config = {"params": None, "score": 0}

    if len(list(itertools.product(*HYPER_PARAMS.values()))) == 1:
        return {k: v[0] for k, v in HYPER_PARAMS.items()}

    print('\t',f"Testing {len(list(itertools.product(*HYPER_PARAMS.values())))} configurations...")

    for params in itertools.product(*HYPER_PARAMS.values()):
        params = {k: v for k, v in zip(HYPER_PARAMS, params)}

        if ((params['PU_k']==3 and params['PU_t'] not in [2/3,1]) or 
            (params['PU_k']==5 and params['PU_t'] not in [4/5,1]) or
            (params['PU_k']==8 and params['PU_t'] not in [4/5,7/8,1])):
            continue

        score = cv_train_with_params(
            x_train, y_train, model, params, verbose=0, random_state=seed
        )

        print('\t\t',f"Score: {score}, Params: {params}")

        if score > best_config["score"]:
            best_config = {"params": params, "score": score}

    return best_config["params"]
