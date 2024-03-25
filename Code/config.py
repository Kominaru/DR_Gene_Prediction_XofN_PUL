import argparse
from typing import Union

def read_config() -> argparse.Namespace:

    """
    Read the configuration parameters from the command line.

    Returns:
        dict: Configuration parameters
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="PathDIP", help="Dataset name")
    parser.add_argument("--classifier", type=str, default="CAT", help="Classifier name")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--cv_outer", type=int, default=10, help="Outer cross-validation folds")
    parser.add_argument("--cv_inner", type=int, default=5, help="Inner cross-validation folds")
    parser.add_argument("--sampling_method", default="normal", help="Sampling method")
    parser.add_argument("--binary_threshold", type=float, default=0.005, help="Binary threshold")
    parser.add_argument("--features_to_use", type=str, default=["original"], help="Features to use")
    parser.add_argument("--xofn_min_sample_leaf", type=int, default=None, help="XofN minimum sample leaf")
    parser.add_argument("--max_xofn_size", type=int, default=None, help="Maximum XofN size")
    parser.add_argument("--pu_learning", type=str, default=False, help="PU learning")
    parser.add_argument("--pu_k", type=int, nargs="+", default=10, help="PU k")
    parser.add_argument("--pu_t", type=float, nargs="+", default=None, help="PU t values")
    parser.add_argument("--pul_num_features", type=int, help="PUL kNN number of features")
    parser.add_argument("--pul_fs", type=str, default="all", help="PUL feature selection")

    args = parser.parse_args()
    args = vars(args)

    return args
