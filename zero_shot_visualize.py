"""
Extra visualizations from the zero-shot baseline results -- reads the
predictions already saved by 03_zero_shot_baseline.py, no re-running the
model needed.

Run this locally:
    python 04_zero_shot_visualize.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

INPUT_PATH = "data/zero_shot_predictions.csv"

drug_colors = {
    "atorvastatin": "#E76F51",
    "ibuprofen":    "#2A9D8F",
    "metformin":    "#264653",
}


def main():
    df = pd.read_csv(INPUT_PATH)
    df["correct"] = df["label"] == df["predicted_label"]
    os.makedirs("results", exist_ok=True)

    # 1. Confidence distribution: correct vs incorrect predictions.
    # A good model should be MORE confident when right, LESS confident when
    # wrong. If the two overlap heavily, the model's confidence score isn't
    # trustworthy as a signal of correctness.
    fig, ax = plt.subplots(figsize=(8, 5))
    for correct, label in [(True, "Correct"), (False, "Incorrect")]:
        subset = df[df["correct"] == correct]
        ax.hist(subset["prediction_confidence"], bins=20, alpha=0.6, label=label)
    ax.set_xlabel("Model confidence")
    ax.set_ylabel("Number of predictions")
    ax.set_title("Zero-shot: confidence when correct vs. incorrect")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/zero_shot_confidence_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()

    # 2. Accuracy per drug -- does the baseline do better or worse for any
    # one of the three drugs?
    per_drug_acc = (df.groupby("queried_drug")["correct"].mean() * 100).round(1)
    print("Accuracy by drug:")
    print(per_drug_acc)

    fig, ax = plt.subplots(figsize=(6, 4))
    per_drug_acc.plot.bar(ax=ax, color=[drug_colors[d] for d in per_drug_acc.index])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Zero-shot accuracy by drug")
    ax.set_xlabel("")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("results/zero_shot_accuracy_by_drug.png", dpi=150, bbox_inches="tight")
    plt.show()

    # 3. Precision / recall / F1 side by side -- makes the class-imbalance
    # problem visible at a glance (the "not serious" bars should look much
    # worse than "serious").
    p, r, f1, _ = precision_recall_fscore_support(df["label"], df["predicted_label"])
    metrics_df = pd.DataFrame(
        {"precision": p, "recall": r, "f1": f1},
        index=["not serious", "serious"],
    )
    print("\nMetrics by class:")
    print(metrics_df.round(3))

    fig, ax = plt.subplots(figsize=(7, 5))
    metrics_df.plot.bar(ax=ax, color=["#264653", "#2A9D8F", "#E76F51"])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Zero-shot baseline: precision, recall, F1 by class")
    plt.xticks(rotation=0)
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig("results/zero_shot_metrics_by_class.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()