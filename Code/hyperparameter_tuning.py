import itertools
from code.model_training import cv_train_with_params

def get_hyperparam_combinations(HYPER_PARAMS):
    """
    Get all possible combinations of hyperparameters.

    Parameters:
        - HYPER_PARAMS (dict): A dictionary containing the hyperparameters to be tuned.

    Returns:
    list: A list of dictionaries containing all possible combinations of hyperparameters.
    """

    temp = HYPER_PARAMS.copy()

    # Transform each value in the dictionary into a list
    for key, value in temp.items():
        if not isinstance(value, list):
            temp[key] = [value]

    return [dict(zip(temp, values)) for values in itertools.product(*temp.values())]


def grid_search_hyperparams(HYPER_PARAMS, x_train, y_train, random_state=42):
    """
    Perform grid search to find the best hyperparameters for a given model.

    Parameters:
        - HYPER_PARAMS (dict): A dictionary containing the hyperparameters to be tuned.
        - x_train (array-like): The input features for training.
        - y_train (array-like): The target variable for training.
        - cv_inner (int, optional): The number of folds for inner cross-validation.

    Returns:
    dict: The best hyperparameters found during grid search.
    """
    best_config = {"params": None, "score": 0}

    hyperparam_combinations = get_hyperparam_combinations(HYPER_PARAMS)

    if len(hyperparam_combinations) == 1:
        return {k: v[0] if isinstance(v,list) else v for k, v in HYPER_PARAMS.items()}

    for params in hyperparam_combinations:

        if ((params['pu_k']==3 and params['pu_t'] not in [.666,1]) or 
            (params['pu_k']==5 and params['pu_t'] not in [4/5,1]) or
            (params['pu_k']==8 and params['pu_t'] not in [3/4,7/8,1])):
            continue

        score = cv_train_with_params(
            x_train, y_train, params["classifier"], params, random_state=random_state, verbose=0
        )

        # print('\t\t',f"Score: {score}, Params: {params}")

        if score > best_config["score"]:
            best_config = {"params": params, "score": score}

    return best_config["params"]
