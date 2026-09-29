# Linear Regression from Scratch: Study Notes & Troubleshooting Guide

Study notes summarizing the linear regression implementation, the debugging steps worked through, and the key concepts resolved along the way.

## 1. Core Implementation & Architecture

- **Dataset Handling:** Loaded tabular student data (`student_exam_performance.csv`) using `pandas` and filtered for numeric columns using `df.select_dtypes(include=[np.number])` to prevent non-numeric processing errors.
- **Feature Engineering (Intercept/Bias Term):** Added a dummy feature column (`X["dummmy"] = 1.0`) to account for the model intercept during matrix operations.
- **Feature Scaling (Standardization):** Standardized features by subtracting the mean and dividing by the standard deviation to stabilize gradient descent and prevent numerical overflow.
- **Loss Function (Gradient Calculation):** Implemented the vectorised gradient calculation:

  $$
  \text{loss} = \frac{1}{2N} X^T (X\theta - y)
  $$

  where $N$ is the number of samples, $X$ is the feature matrix, $\theta$ (`parameters`) is the weight vector, and $y$ is the target array.

## 2. Key Doubts & Resolved Issues

### Issue A: Iteration Type Error (`TypeError: 'int' object is not iterable`)

- **The Problem:** Attempting to loop directly over an integer (`for i in iterations:`) where `iterations = 500`.
- **The Resolution:** Wrapped the integer inside Python's `range()` function (`for i in range(iterations):`) to generate an iterable sequence of loop steps.

### Issue B: Non-Numeric Data & `NaN` Propagation

- **The Problem:**
  1. Unfiltered categorical columns (`gender`, `education_level`, etc.) caused `TypeError: can't multiply sequence by non-int of type 'float'`.
  2. Missing values (`NaN`) in columns like `previous_gpa` resulted in `NaN` means/standard deviations, breaking model parameters and rendering predictions blank.
- **The Resolution:**
  1. Used `df.select_dtypes(include=[np.number])` to isolate numerical features.
  2. Applied `.fillna(0)` before performing standardization to clean missing values safely.

### Issue C: Gradient Explosion & Divergence

- **The Problem:** Running gradient descent on unscaled features with large value ranges caused gradients to blow up infinitely, returning `NaN` weights.
- **The Resolution:** Standardized features to a mean of $0$ and standard deviation of $1$, and lowered the learning rate to ensure smooth convergence.

### Issue D: NumPy 1D Array Transpose Behavior

- **The Problem:** Wondering why `np.transpose()` was used inside the loss function.
- **The Resolution:** While standard linear algebra uses row/column vector transposes (e.g., $e^T e$), NumPy treats 1D arrays as directionless, meaning `np.transpose()` has no effect on them. However, dot products like `np.dot(a, a)` correctly compute the sum of squared elements for 1D arrays natively.
