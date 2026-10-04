# K-Nearest Neighbors (KNN) from Scratch in Python

A clean, from-scratch implementation of the K-Nearest Neighbors (KNN) classification algorithm using `NumPy`, `Pandas`, and `Scikit-Learn` (for dataset splitting only). This project avoids high-level machine learning libraries for the core KNN mechanism, focusing instead on manual distance computation, neighbor ranking, and majority voting.

## Project Structure

```
knn/
│
├── KNNAlgorithmDataset.csv   # Dataset containing features and binary diagnostic classes
├── main.py                   # Main script for data preprocessing, KNN evaluation, and accuracy calculation
└── README.md                 # Project documentation and study notes
```

## Features & Implementation Details

1. **Data Preprocessing & Cleaning:**
   * Loads tabular data using `Pandas`.
   * Drops unnecessary columns such as `id` and empty/formatting columns (`Unnamed: 32`).
   * Encodes diagnostic string labels into binary integer values (`"M"` $\rightarrow 1$, `"B"` $\rightarrow 0$).

2. **Data Splitting:**
   * Uses `train_test_split` from `sklearn.model_selection` to separate data into training and testing subsets (70% train, 30% test).
   * Converts Pandas DataFrames into high-performance `NumPy` arrays (`.to_numpy()`) for efficient vector operations.

3. **Euclidean Distance Metric:**
   * Calculates the straight-line distance between feature vectors using NumPy:
     $$\text{Distance} = \sqrt{\sum (p_1 - p_2)^2}$$

4. **KNN Prediction Pipeline:**
   * Iterates through each test sample and computes its Euclidean distance to all training points.
   * Sorts distances using `np.argsort()` and extracts the indices of the $K$ nearest neighbors ($K = 3$).
   * Performs a **majority vote** on the neighbor classes to predict the final label for the test point.

5. **Model Evaluation:**
   * Compares predicted labels against true test labels (`y_test`) to compute overall classification accuracy.

## Requirements

Ensure you have the following Python libraries installed in your active environment:

```bash
pip install numpy pandas scikit-learn
```

## How to Run

1. Open your terminal and navigate to your project directory.
2. Execute the script:

```bash
python main.py
```

## Complete Code Reference (`main.py`)

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Load Data
data = pd.read_csv("KNNAlgorithmDataset.csv")
data.drop(["id", "Unnamed: 32"], axis=1, inplace=True, errors='ignore')

k = 3

# Encode labels: Malignant (M) = 1, Benign = 0
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x = data.drop(["diagnosis"], axis=1)

# 2. Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

x_test, x_train = x_test.to_numpy(), x_train.to_numpy() 

# 3. Euclidean Distance Function
def euclideanDistance(point1, point2):
    dist = np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))
    return dist

# 4. KNN Prediction Loop
y_pred = []
for point1 in x_test:
    dist = []
    for point2 in x_train:
        dist.append(euclideanDistance(point1, point2))

    # Find indices of K nearest neighbors
    points = np.argsort(dist)[:k]
    pointClass = [y_train[i] for i in points]
    
    # Majority voting between class 1 and class 0
    count1 = pointClass.count(1)
    count0 = pointClass.count(0)

    if count1 > count0:
        y_pred.append(1)
    else:
        y_pred.append(0)

# 5. Accuracy Metric
accuracy = np.mean(y_test == np.array(y_pred))
print(f"KNN Accuracy: {accuracy:.4f}")
```