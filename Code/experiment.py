import neptune
import numpy as np
from code.config import read_config
from code.main import run_experiment

PARAMS = read_config()
SEEDS = [14, 33, 39, 42, 727, 1312, 1337, 56709, 177013, 241543903]

neptune_run = neptune.init_run(project='JorgePRuza-Tesis/DR-Gene-Prediction', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MDE0YzVjYi1hODRmLTQ4M2YtYTA0NC1mYzNjNDc5YTRlOGQifQ==')

neptune_run["parameters"] = neptune.utils.stringify_unsupported(PARAMS)

PARAMS["neptune_run"] = neptune_run

run_metrics_list = []

for seed in SEEDS:
    run_metrics = run_experiment(PARAMS, random_state=seed, neptune_run=neptune_run)
    run_metrics_list.append(run_metrics)

for metric in run_metrics_list[0][0].keys():
    neptune_run["metrics/avg/test/" + metric] = np.mean([np.mean([fold[metric] for fold in run_metrics]) for run_metrics in run_metrics_list])

neptune_run.stop()
