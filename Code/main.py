import os
from codecarbon import EmissionsTracker
import neptune
import numpy as np
from code.config import read_config
from code.experiment import run_experiment

if __name__ == "__main__":

    tracker = EmissionsTracker(tracking_mode="process", save_to_file=False, log_level="error")
    tracker.start()

    SEEDS = [14, 33, 39, 42, 727, 1312, 1337, 56709, 177013, 241543903]

    args = read_config()

    if args["neptune"]:
        neptune_run = neptune.init_run(
            project="JorgePRuza-Tesis/DR-Gene-Prediction",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MDE0YzVjYi1hODRmLTQ4M2YtYTA0NC1mYzNjNDc5YTRlOGQifQ==",
        )
        neptune_run["parameters"] = neptune.utils.stringify_unsupported(args)

    run_metrics_list = []
    run_preds_list = []

    search_space = {
        "pu_k": args["pu_k"],
        "pu_t": args["pu_t"],
    }

    for seed in SEEDS:
        run_metrics, run_preds = run_experiment(
            args["dataset"],
            args["classifier"],
            args["pul_num_features"],
            args["pu_learning"],
            search_space,
            random_state=seed,
            neptune_run=neptune_run if args["neptune"] else None,
            tracker=tracker,
        )
        run_metrics_list.append(run_metrics)
        run_preds_list.append(run_preds)

    for metric in run_metrics_list[0][0].keys():
        if args["neptune"]:
            neptune_run["metrics/avg/test/" + metric] = np.mean(
                [np.mean([fold[metric] for fold in run_metrics]) for run_metrics in run_metrics_list]
            )
        else:
            print(
                f"metrics/avg/test/{metric}: {np.mean([np.mean([fold[metric] for fold in run_metrics]) for run_metrics in run_metrics_list])}"
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

    tracker.stop()

    print("Total emissions: ", tracker.final_emissions * 1000 / len(SEEDS), " gCO2e")

    if args["neptune"]:
        neptune_run["predictions/avg"].upload("avg_probs.csv")
        neptune_run.stop()

    if os.path.exists("preds_temp.csv"):
        os.remove("preds_temp.csv")
    if os.path.exists("preds_temp.csv"):
        os.remove("avg_probs.csv")
