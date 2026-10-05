"""Test the project's Gini-based root split and plot classification metrics."""

from pathlib import Path
from urllib.parse import urlparse
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = Path(__file__).with_name("phishing_site_urls.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.20
SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure", "account",
    "bank", "confirm", "signin", "password", "paypal",
]


def url_to_features(url: str) -> dict[str, int]:
    """Extract the same numeric URL features as main.py."""
    url = str(url).strip()
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        host = parsed.netloc.lower()
    except ValueError:
        # Some noisy dataset URLs contain malformed bracketed IPv6 authorities.
        host = re.split(r"[/\\?#]", url.split("://", 1)[-1], maxsplit=1)[0].lower()
    return {
        "url_length": len(url),
        "host_length": len(host),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_digits": sum(character.isdigit() for character in url),
        "num_slashes": url.count("/"),
        "has_at": int("@" in url),
        "has_ip": int(bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}(:\d+)?", host))),
        "uses_https": int(url.lower().startswith("https")),
        "num_subdomains": max(host.count(".") - 1, 0),
        "num_suspicious_words": sum(word in url.lower() for word in SUSPICIOUS_WORDS),
        "has_port": int(":" in host),
    }


def find_best_split(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[int, float, float]:
    """Return the feature, threshold, and weighted Gini of the best stump split."""
    best_feature = -1
    best_threshold = 0.0
    best_weighted_gini = float("inf")
    n_samples = len(y)

    for feature_index in range(X.shape[1]):
        values = X[:, feature_index]
        # Only test boundaries between distinct sorted values.
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        sorted_y = y[order]
        boundaries = np.flatnonzero(sorted_values[:-1] < sorted_values[1:]) + 1
        if boundaries.size == 0:
            continue

        cumulative_positive = np.cumsum(sorted_y == 1)
        left_positive = cumulative_positive[boundaries - 1]
        left_negative = boundaries - left_positive
        right_size = n_samples - boundaries
        right_positive = cumulative_positive[-1] - left_positive
        right_negative = right_size - right_positive

        left_gini_weighted = 2 * left_positive * left_negative / boundaries
        right_gini_weighted = 2 * right_positive * right_negative / right_size
        weighted_gini = (left_gini_weighted + right_gini_weighted) / n_samples
        best_index = int(np.argmin(weighted_gini))

        if weighted_gini[best_index] < best_weighted_gini:
            boundary = boundaries[best_index]
            best_feature = feature_index
            best_threshold = float(
                (sorted_values[boundary - 1] + sorted_values[boundary]) / 2
            )
            best_weighted_gini = float(weighted_gini[best_index])

    if best_feature < 0:
        raise ValueError("No valid split was found for the training data.")
    return best_feature, best_threshold, best_weighted_gini


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    if not {"URL", "Label"}.issubset(data.columns):
        raise ValueError("Dataset must have 'URL' and 'Label' columns.")
    data = data.dropna(subset=["URL", "Label"])
    labels = data["Label"].astype(str).str.strip().str.lower()
    label_map = {"bad": 1, "good": 0}
    y_series = labels.map(label_map)
    if y_series.isna().any():
        raise ValueError("Labels must be 'bad' or 'good'.")

    features = pd.DataFrame([url_to_features(url) for url in data["URL"]])
    feature_names = features.columns.tolist()
    X = features.to_numpy(dtype=float)
    y = y_series.to_numpy(dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    feature_index, threshold, weighted_gini = find_best_split(X_train, y_train)
    y_train_left = y_train[X_train[:, feature_index] <= threshold]
    y_train_right = y_train[X_train[:, feature_index] > threshold]
    # Label each leaf by its training-set majority class; scores use its phishing rate.
    left_probability = float(np.mean(y_train_left == 1))
    right_probability = float(np.mean(y_train_right == 1))
    left_label = int(left_probability >= 0.5)
    right_label = int(right_probability >= 0.5)

    test_left = X_test[:, feature_index] <= threshold
    y_score = np.where(test_left, left_probability, right_probability)
    y_pred = np.where(test_left, left_label, right_label)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_score),
    }
    print("Decision stump test-set evaluation (positive class: phishing)")
    print(f"Best split: {feature_names[feature_index]} <= {threshold:.2f}")
    print(f"Weighted Gini impurity: {weighted_gini:.4f}")
    print(f"Training samples: {len(y_train):,} | Test samples: {len(y_test):,}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred, labels=[0, 1], target_names=["Legitimate", "Phishing"],
        zero_division=0,
    ))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    fig.suptitle("Phishing URL Decision Stump — Test Set", fontsize=16)
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=[0, 1], display_labels=["Legitimate", "Phishing"],
        cmap="Blues", colorbar=False, values_format="d", ax=axes[0, 0],
    )
    axes[0, 0].set_title("Confusion Matrix")

    bars = axes[0, 1].bar(
        list(metrics), list(metrics.values()),
        color=["#2563eb", "#0d9488", "#e11d48", "#7c3aed", "#64748b"],
    )
    axes[0, 1].bar_label(bars, fmt="%.3f", padding=3)
    axes[0, 1].set_ylim(0, 1.12)
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_title("Classification Metrics")
    axes[0, 1].tick_params(axis="x", rotation=20)

    RocCurveDisplay.from_predictions(y_test, y_score, name="Decision stump", ax=axes[1, 0])
    axes[1, 0].plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    axes[1, 0].set_title("ROC Curve (Phishing)")
    axes[1, 0].legend(loc="lower right")

    branch_counts = np.array([
        [np.sum(y_train_left == 0), np.sum(y_train_left == 1)],
        [np.sum(y_train_right == 0), np.sum(y_train_right == 1)],
    ])
    positions = np.arange(2)
    axes[1, 1].bar(positions, branch_counts[:, 0], label="Legitimate", color="#60a5fa")
    axes[1, 1].bar(
        positions, branch_counts[:, 1], bottom=branch_counts[:, 0],
        label="Phishing", color="#fb7185",
    )
    axes[1, 1].set_xticks(positions, ["≤ threshold", "> threshold"])
    axes[1, 1].set_ylabel("Training URLs")
    axes[1, 1].set_title(f"Training Samples by {feature_names[feature_index]}")
    axes[1, 1].legend()

    output_path = Path(__file__).with_name("decision_tree_evaluation.png")
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"\nVisualization saved to: {output_path}")
    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
