import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
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



data = pd.read_csv(Path(__file__).with_name("KNNAlgorithmDataset.csv"))

data.drop(["id","Unnamed: 32"],axis=1,inplace=True)


k = 5

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x = data.drop(["diagnosis"],axis=1)


x_train , x_test , y_train , y_test  = train_test_split(x,y,test_size = 0.3,random_state=1)

x_test , x_train  = x_test.to_numpy()  , x_train.to_numpy() 
def euclideanDistance(point1,point2) :
    dist = np.sqrt(np.sum( ( np.array(point1) - np.array(point2) ) **2 ))
    return dist

y_pred = []
y_score = []
for point1 in x_test:
    dist = []
    for point2 in x_train:
        dist.append(euclideanDistance(point1,point2))

    points = np.argsort(dist)[:k]
    pointClass = [y_train[i] for i in points]
    count1 = pointClass.count(1)
    count2 = pointClass.count(0)
    # The malignant-neighbor fraction provides a score for the ROC curve.
    y_score.append(count1 / k)

    if count1 > count2 :
        y_pred.append(1)
    else:
        y_pred.append(0)


metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1": f1_score(y_test, y_pred, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, y_score),
}

print(f"KNN evaluation (k={k}, positive class: Malignant)")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")
print("\nClassification report:")
print(classification_report(
    y_test, y_pred, labels=[0, 1],
    target_names=["Benign", "Malignant"], zero_division=0,
))

fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
fig.suptitle(f"KNN Test Set Evaluation (k={k})")

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, labels=[0, 1], display_labels=["Benign", "Malignant"],
    cmap="Blues", colorbar=False, ax=axes[0],
)
axes[0].set_title("Confusion Matrix")

bars = axes[1].bar(metrics.keys(), metrics.values(), color=[
    "#2563eb", "#0d9488", "#e11d48", "#7c3aed", "#64748b",
])
axes[1].bar_label(bars, fmt="%.3f", padding=3)
axes[1].set_ylim(0, 1.12)
axes[1].set_ylabel("Score")
axes[1].set_title("Classification Metrics")
axes[1].tick_params(axis="x", rotation=30)

RocCurveDisplay.from_predictions(
    y_test, y_score, name="KNN", ax=axes[2],
)
axes[2].plot([0, 1], [0, 1], "--", color="gray", label="Chance")
axes[2].set_title("ROC Curve (Malignant)")
axes[2].legend(loc="lower right")

output_path = Path(__file__).with_name("knn_evaluation.png")
fig.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Visualization saved to: {output_path}")
plt.show()
