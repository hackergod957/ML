# Logistic Regression from Scratch in Python

A clean, from-scratch implementation of binary logistic regression using `NumPy`, `Pandas`, and `Matplotlib`. This project is built without relying on high-level machine learning libraries like `scikit-learn` for the core model training, allowing for a deep understanding of gradient descent, the sigmoid function, and cross-entropy loss.

---

## Project Structure

```text
logistic regression/
│
├── breast_cancer.csv      # Dataset containing features and target binary classes
├── main.py                # Main script for data preprocessing, training, evaluation, and plotting
└── README.md              # Project documentation and study notes
```

---

## Features & Implementation Details

1. **Data Preprocessing & Standardization:**

   * Automatically isolates numeric features from the dataset.
   * Standardizes feature columns to a mean of $0$ and a standard deviation of $1$ to stabilize gradient descent and prevent numerical overflow.
   * Appends a bias/intercept term column (`dummy`) filled with ones.
2. **Sigmoid Activation Function:**

   * Maps linear combinations of features to probabilities between $0$ and $1$:
     $$
     \sigma(z) = \frac{1}{1 + e^{-z}}
     $$
3. **Binary Cross-Entropy Loss Function:**

   * Calculates the cost function with numerical stability protection (using an epsilon clip to prevent $\log(0)$ errors):
     $$
     \text{Loss} = -\frac{1}{N} \sum \left[ y \log(p) + (1 - y) \log(1 - p) \right]
     $$
4. **Gradient Descent Optimization:**

   * Iteratively updates weights ($\theta$) using the computed gradient vector:
     $$
     \theta \leftarrow \theta - \alpha \cdot \frac{1}{N} X^T (h - y)
     $$
5. **Evaluation Metrics:**

   * Computes **Accuracy**, **Precision**, **Recall**, **F1-Score**, and constructs a full **Confusion Matrix** (True Negatives, False Positives, False Negatives, True Positives).
6. **Visualization Dashboard:**

   * Generates a 2-panel subplot showing:
     * **Training Loss Curve:** Tracking cross-entropy loss reduction across iterations.
     * **Confusion Matrix Heatmap:** Visual representation of model classification performance.

---

## Requirements

Ensure you have the following Python libraries installed in your active environment:

```bash
pip install -r requirements.txt
```

---

## How to Run

1. Open your terminal and navigate to the project directory:
   ```bash
   cd ~/Desktop/ml/logistic\ regression
   ```
2. Execute the main script:
   ```python
   python main.py
   ```

---
