"""
Step: Zero-shot baseline classification.

Classifies each report's reaction text as "serious" or "not serious" using
a pretrained model with NO training on our data -- this is the baseline to
beat with the fine-tuned model later.

Run this locally:
    python 03_zero_shot_baseline.py

Requires (install locally):
    uv add transformers torch tqdm scikit-learn

First run downloads the model (~1.6GB), so it needs internet access and
will be slow the first time regardless of your machine.
"""

import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch

INPUT_PATH = "data/model_ready.csv"
OUTPUT_PATH = "data/zero_shot_predictions.csv"

CANDIDATE_LABELS = ["serious adverse event", "non-serious adverse event"]
LABEL_TO_INT = {"serious adverse event": 1, "non-serious adverse event": 0}

SAMPLE_SIZE_FOR_TIMING = 20


def get_device():
    """Use Apple Silicon GPU (MPS) if available, else CPU."""
    if torch.backends.mps.is_available():
        print("Using Apple Silicon GPU (MPS) -- should be noticeably faster than CPU.")
        return "mps"
    print("No GPU acceleration available -- running on CPU. This will be slow.")
    return "cpu"


def estimate_runtime(classifier, texts, sample_size):
    """Time a small sample and extrapolate to the full dataset."""
    sample = texts[:sample_size]
    print(f"\nTiming a sample of {sample_size} examples...")
    start = time.time()
    for text in sample:
        classifier(text, CANDIDATE_LABELS)
    elapsed = time.time() - start

    per_example = elapsed / sample_size
    total_estimate_minutes = (per_example * len(texts)) / 60

    print(f"  {elapsed:.1f}s for {sample_size} examples ({per_example:.2f}s/example)")
    print(f"  Estimated time for all {len(texts)} examples: {total_estimate_minutes:.1f} minutes")
    return total_estimate_minutes


def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} examples")

    device = get_device()
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

    texts = df["reaction_text"].tolist()

    estimate = estimate_runtime(classifier, texts, SAMPLE_SIZE_FOR_TIMING)
    print(f"\nEstimated {estimate:.0f} minutes for the full dataset. Running now...")

    print("\nRunning zero-shot classification on the full dataset...")
    predictions = []
    scores = []
    start = time.time()

    for i, text in enumerate(texts):
        result = classifier(text, CANDIDATE_LABELS)
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        predictions.append(LABEL_TO_INT[top_label])
        scores.append(top_score)

        if (i + 1) % 200 == 0:
            elapsed_min = (time.time() - start) / 60
            print(f"  {i + 1}/{len(texts)} done ({elapsed_min:.1f} min elapsed)")

    df["predicted_label"] = predictions
    df["prediction_confidence"] = scores
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved predictions to {OUTPUT_PATH}")

    print("\n" + "=" * 60)
    print("ZERO-SHOT BASELINE RESULTS")
    print("=" * 60)
    print(classification_report(df["label"], df["predicted_label"],
                                 target_names=["not serious", "serious"]))

    cm = confusion_matrix(df["label"], df["predicted_label"])
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(cm)

    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["not serious", "serious"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Zero-shot baseline: confusion matrix")
    plt.tight_layout()
    plt.savefig("results/zero_shot_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved chart to results/zero_shot_confusion_matrix.png")


if __name__ == "__main__":
    main()