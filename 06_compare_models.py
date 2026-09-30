import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"

# (precision, recall, f1) per model per class, from the confirmed
# confusion matrices on the shared 1,200-row held-out test set:
#   zero-shot   [[24, 230], [10, 936]]
#   fine-tuned  [[181, 73], [67, 879]]
comparison = pd.DataFrame({
    ("Zero-shot", "not serious"):  [0.71, 0.09, 0.16],
    ("Zero-shot", "serious"):      [0.80, 0.99, 0.88],
    ("Fine-tuned", "not serious"): [0.73, 0.71, 0.72],
    ("Fine-tuned", "serious"):     [0.92, 0.93, 0.93],
}, index=["precision", "recall", "f1"]).T


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    comparison.plot.bar(ax=ax, width=0.75, color=["#264653", "#2A9D8F", "#E76F51"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Zero-shot vs fine-tuned: same held-out test set (n=1,200)")
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/zero_shot_vs_fine_tuned.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(comparison.round(2))


if __name__ == "__main__":
    main()