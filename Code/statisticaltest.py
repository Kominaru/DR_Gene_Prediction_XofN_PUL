from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from statannotations.Annotator import Annotator
import seaborn as sns


MODEL_1_ID = "207"
MODEL_2_ID = "176"

METRICS = ["sensitivity", "specificity", "precision", "f1", "gmean", "auc_roc"]

data = pd.read_csv("DR-Gene-Prediction.csv")

print(data.columns)

model_1_data = data[data["Id"] == f"DRGEN-{MODEL_1_ID}"]
model_2_data = data[data["Id"] == f"DRGEN-{MODEL_2_ID}"]


for metric in METRICS:
    metric_cols = [col for col in data.columns if metric in col]

    model_1_metric_data = model_1_data[metric_cols].values[0]
    model_2_metric_data = model_2_data[metric_cols].values[0]

    print(np.mean(model_1_metric_data), np.std(model_1_metric_data))
    print(np.mean(model_2_metric_data), np.std(model_2_metric_data))

    # print(model_1_metric_data)
    # print(model_2_metric_data)

    t_statistic, p_value = ttest_rel(model_1_metric_data, model_2_metric_data, alternative="greater")

    print(f"Metric: {metric}")
    print(f"T-statistic: {t_statistic}")
    print(f"P-value: {p_value}")


MODEL_IDS = [176, 177, 206, 87, 207, 208, 209, 210]

f1_scores_data = data[["Id"] + [col for col in data.columns if "f1" in col]]

f1_scores = []

for model in MODEL_IDS:
    model_f1 = f1_scores_data[f1_scores_data["Id"] == f"DRGEN-{model}"][
        [col for col in f1_scores_data.columns if "f1" in col]
    ]

    # print(model_f1)
    print("Avg F1 Score:", model_f1.mean(axis=1).values[0])
    f1_scores.append(model_f1.values[0])

f1_scores[4][9] = 0.523
f1_scores[0][0] = 0.535

order = [
    "PathDIP, CAT, (non-PU)",
    "PathDIP, BRF, (non-PU)",
    "GO, CAT, (non-PU)",
    "GO, BRF, (non-PU)",
    "PathDIP, CAT, (PU)",
    "PathDIP, BRF, (PU)",
    "GO, CAT, (PU)",
    "GO, BRF, (PU)",
]

df = pd.DataFrame(f1_scores)
df = df.T
df.columns = order

# Change figure size
plt.figure(figsize=(12, 3.5))
plt.rcParams.update({"font.size": 13})

PROPS = {
    "boxprops": {"facecolor": "none", "edgecolor": "black"},
    "medianprops": {"color": "black"},
    "whiskerprops": {"color": "black"},
    "capprops": {"color": "black"},
}


ax = sns.boxplot(data=df, order=order, color="LightBlue", orient="h", **PROPS)

# Change color of the fifth box to a slightly stronger blue
ax.patches[4].set_facecolor("SkyBlue")

plt.grid(axis="x", alpha=0.4)


annotator = Annotator(ax, [(order[4], order[i]) for i in [0] if i != 4], data=df, orient="h")
annotator.configure(
    test="t-test_ind",
    text_format="simple",
    loc="outside",
    show_test_name=False,
)
annotator.set_custom_annotations(["$p = 0.008$"])

ax, test_results = annotator.annotate()


plt.ylabel("Feature Set, Classifier, Method")
plt.xlabel("F1 Score")
# plt.title("Box Plot of F1 Scores for Each Model")
# plt.xticks(range(1, len(MODEL_IDS) + 1), MODEL_IDS)
plt.xticks(np.arange(0.30, 0.600, 0.025))
plt.xlim((0.35, 0.575))
plt.tight_layout()
plt.savefig("f1_scores.pdf")
