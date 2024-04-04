import neptune
import numpy as np
from code.config import read_config
from code.main import run_experiment

PARAMS = read_config()
SEEDS = [14, 33, 39, 42, 727, 1312, 1337, 56709, 177013, 241543903]

neptune_run = neptune.init_run(
    project="JorgePRuza-Tesis/DR-Gene-Prediction",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MDE0YzVjYi1hODRmLTQ4M2YtYTA0NC1mYzNjNDc5YTRlOGQifQ==",
)

neptune_run["parameters"] = neptune.utils.stringify_unsupported(PARAMS)

PARAMS["neptune_run"] = neptune_run

run_metrics_list = []
run_preds_list = []

for seed in SEEDS:
    run_metrics, run_preds = run_experiment(PARAMS, random_state=seed, neptune_run=neptune_run)
    run_metrics_list.append(run_metrics)
    run_preds_list.append(run_preds)

for metric in run_metrics_list[0][0].keys():
    neptune_run["metrics/avg/test/" + metric] = np.mean(
        [np.mean([fold[metric] for fold in run_metrics]) for run_metrics in run_metrics_list]
    )

# Get the average prediction for each gene
for i in range(1, len(run_preds_list)):
    run_preds_list[i] = run_preds_list[i].sort_values(by="gene")

avg_preds = run_preds_list[0].copy()

for run_preds in run_preds_list[1:]:
    avg_preds["prob"] += run_preds["prob"]

avg_preds["prob"] /= len(run_preds_list)

avg_preds = avg_preds.sort_values(by="prob", ascending=False)

# Drop the id column
avg_preds = avg_preds.drop(columns=["id"])

# Save the predictions to neptune
avg_preds.to_csv("avg_probs.csv", index=False)
neptune_run["predictions/avg"].upload("avg_probs.csv")

neptune_run.stop()
