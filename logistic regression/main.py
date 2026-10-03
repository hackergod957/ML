import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

df = pd.read_csv("breast_cancer.csv")
df['Class'] = df['Class'].replace({2: 0, 4: 1})
df.head(5)


X = df.drop(columns="Class")
Y = df[["Class"]].copy().to_numpy().flatten()
X["dummy"] = np.ones(X.shape[0])

def LossFunction(p,y):
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    loss = (-1/y.shape[0]) * (np.dot(y,np.log(p)) + np.dot(1-y,np.log(1-p)))
    return loss

para = np.zeros(X.shape[1])

iterations = 10000
lrate = 0.01
train_losses = []

for i in range(iterations):
    z = np.dot(X,para)
    h = 1 / (1 +  np.exp(-z ))
    h = h.flatten()
    loss = LossFunction(h,Y)
    train_losses.append(loss)

    para -=  lrate * (1/X.shape[0]) * np.dot(np.transpose(X),h-Y)

print("Training complete! Final loss:", loss)

y_pred = (h >= 0.5).astype(int)

accuracy = np.mean(y_pred == Y)
tp = np.sum((y_pred == 1) & (Y == 1))
fp = np.sum((y_pred == 1) & (Y == 0))
fn = np.sum((y_pred == 0) & (Y == 1))
tn = np.sum((y_pred == 0) & (Y == 0))

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("--- Evaluation Metrics (Full Dataset) ---")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"Confusion Matrix:\n [[TN: {tn}, FP: {fp}],\n  [FN: {fn}, TP: {tp}]]")

# 5. Visualization Dashboard (Loss Curve & Confusion Matrix)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Training Loss Curve
axes[0].plot(train_losses, color='purple', lw=2)
axes[0].set_title("Training Loss Curve")
axes[0].set_xlabel("Iterations")
axes[0].set_ylabel("Cross-Entropy Loss")
axes[0].grid(True, linestyle='--', alpha=0.6)

# Plot 2: Confusion Matrix Heatmap
cm = np.array([[tn, fp], [fn, tp]])
cax = axes[1].matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
fig.colorbar(cax, ax=axes[1])
axes[1].set_title("Confusion Matrix", y=1.12)
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['Benign (0)', 'Malignant (1)'])
axes[1].set_yticklabels(['Benign (0)', 'Malignant (1)'])

# Annotate confusion matrix numbers
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14, color='black' if cm[i,j] < cm.max()/2 else 'white')

plt.tight_layout()
plt.show()
