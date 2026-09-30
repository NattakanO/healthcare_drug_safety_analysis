import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer,
)
from datasets import Dataset

MODEL_READY_PATH = "data/model_ready.csv"
ZERO_SHOT_PREDS_PATH = "data/zero_shot_predictions.csv"
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "distilbert_finetuned"
RESULTS_DIR = "results"

RANDOM_STATE = 42
TEST_SIZE = 0.2
NUM_EPOCHS = 3


def get_device():
    if torch.backends.mps.is_available():
        print("Using Apple Silicon GPU (MPS).")
        return "mps"
    print("No GPU acceleration available -- using CPU. This will take a while.")
    return "cpu"


class WeightedTrainer(Trainer):
    """A Trainer that applies class weights to the loss, so the model is
    penalized more for getting the minority class ('not serious') wrong,
    instead of defaulting toward the majority class like the zero-shot
    baseline did."""

    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def tokenize_dataset(df, tokenizer):
    ds = Dataset.from_pandas(df[["reaction_text", "label"]].reset_index(drop=True))
    return ds.map(
        lambda batch: tokenizer(batch["reaction_text"], truncation=True, padding="max_length", max_length=64),
        batched=True,
    )


def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["not serious", "serious"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{filename}", dpi=150, bbox_inches="tight")
    plt.show()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_data = pd.read_csv(MODEL_READY_PATH)
    print(f"Loaded {len(model_data)} examples")

    train_df, test_df = train_test_split(
        model_data, test_size=TEST_SIZE, stratify=model_data["label"], random_state=RANDOM_STATE
    )
    print(f"Train: {len(train_df)} | Test (held out): {len(test_df)}")

    class_weights = compute_class_weight(
        "balanced", classes=np.array([0, 1]), y=train_df["label"]
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Class weights (not serious, serious): {class_weights.tolist()}")

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

    train_ds = tokenize_dataset(train_df, tokenizer)
    test_ds = tokenize_dataset(test_df, tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="no",   # keep this simple -- we save our own metrics/charts below
        logging_steps=50,
        report_to="none",
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
    )

    print("\nTraining...")
    trainer.train()

    print("\nEvaluating fine-tuned model on the held-out test set...")
    predictions = trainer.predict(test_ds)
    fine_tuned_preds = np.argmax(predictions.predictions, axis=1)

    print("\n" + "=" * 60)
    print("FINE-TUNED MODEL RESULTS (held-out test set)")
    print("=" * 60)
    print(classification_report(test_df["label"], fine_tuned_preds,
                                 target_names=["not serious", "serious"]))
    plot_confusion_matrix(test_df["label"], fine_tuned_preds,
                           "Fine-tuned model: confusion matrix",
                           "fine_tuned_confusion_matrix.png")

    # Fair comparison: re-score the EXISTING zero-shot predictions, but only
    # on this same held-out test set -- not the full 6,000 it was run on
    # before, so both models are judged on identical, unseen-by-training data.
    zero_shot_preds = pd.read_csv(ZERO_SHOT_PREDS_PATH)
    zs_on_test = zero_shot_preds.merge(
        test_df[["safetyreportid", "queried_drug"]],
        on=["safetyreportid", "queried_drug"],
    )
    print(f"\nMatched {len(zs_on_test)} of {len(test_df)} test rows to existing zero-shot predictions")

    print("\n" + "=" * 60)
    print("ZERO-SHOT BASELINE RESULTS (SAME held-out test set, for fair comparison)")
    print("=" * 60)
    print(classification_report(zs_on_test["label"], zs_on_test["predicted_label"],
                                 target_names=["not serious", "serious"]))
    plot_confusion_matrix(zs_on_test["label"], zs_on_test["predicted_label"],
                           "Zero-shot baseline: confusion matrix (held-out test set)",
                           "zero_shot_confusion_matrix_test_set.png")

    # Save the fine-tuned predictions for later use (e.g. dashboard)
    test_df = test_df.copy()
    test_df["fine_tuned_prediction"] = fine_tuned_preds
    test_df.to_csv("data/fine_tuned_predictions.csv", index=False)
    print("\nSaved fine-tuned predictions to data/fine_tuned_predictions.csv")


if __name__ == "__main__":
    main()