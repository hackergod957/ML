# Student Exam Score Prediction using Linear Regression

A machine learning project that predicts students' exam scores using **Multiple Linear Regression implemented from scratch with NumPy**.

The main goal of this project is to understand the mathematics and implementation behind linear regression and gradient descent instead of relying on a pre-built machine learning model.

---

## 📌 Project Overview

This project uses student performance data to predict the final `exam_score` from the numerical features available in the dataset.

The model is trained using **Batch Gradient Descent** to minimize the squared-error loss function.

### Workflow

1. Load the student performance dataset.
2. Select numerical features.
3. Remove `exam_score` from the input features.
4. Handle missing values.
5. Standardize the input features.
6. Add an intercept/bias term.
7. Initialize model parameters.
8. Train the model using gradient descent.
9. Generate predicted exam scores.
10. Evaluate the model using MSE and RMSE.
11. Visualize actual vs. predicted scores.

---

## 🧠 Mathematics

### 1. Linear Regression Model

The model predicts the exam score using:

$$
\hat{y} = X\theta
$$

where:

- $\hat{y}$ = predicted exam scores
- $X$ = feature matrix
- $\theta$ = parameter vector

For an individual example:

$$
\hat{y}^{(i)}
=
\theta_0
+
\theta_1x_1^{(i)}
+
\theta_2x_2^{(i)}
+\cdots+
\theta_nx_n^{(i)}
$$

The parameter $\theta_0$ represents the intercept.

---

### 2. Loss Function

The model minimizes the squared error between predicted and actual scores.

The objective function is:

$$
J(\theta)
=
\frac{1}{2m}
\sum_{i=1}^{m}
\left(
\hat{y}^{(i)}-y^{(i)}
\right)^2
$$

where:

- $J(\theta)$ = cost function
- $m$ = number of training examples
- $\hat{y}^{(i)}$ = predicted score for example $i$
- $y^{(i)}$ = actual score for example $i$

The factor $\frac{1}{2}$ is included to simplify the derivative.

---

### 3. Gradient

The gradient of the cost function with respect to the parameter vector is:

$$
\nabla J(\theta)
=
\frac{1}{m}
X^T
(X\theta-y)
$$

The gradient tells us how the loss changes with respect to each parameter.

---

### 4. Gradient Descent

The parameters are updated iteratively using:

$$
\theta
\leftarrow
\theta
-
\alpha\nabla J(\theta)
$$

where:

- $\theta$ = model parameters
- $\alpha$ = learning rate
- $\nabla J(\theta)$ = gradient of the cost function

Substituting the gradient:

$$
\theta
\leftarrow
\theta
-
\frac{\alpha}{m}
X^T(X\theta-y)
$$

This process is repeated for a fixed number of iterations until the parameters converge toward values that minimize the loss.

---

### 5. Prediction

Once the model has been trained, predictions are calculated using:

$$
\hat{y}=X\theta
$$

---

### 6. Mean Squared Error

The model is evaluated using Mean Squared Error:

$$
MSE
=
\frac{1}{m}
\sum_{i=1}^{m}
\left(
\hat{y}^{(i)}-y^{(i)}
\right)^2
$$

A lower MSE indicates that the predictions are, on average, closer to the actual values.

---

### 7. Root Mean Squared Error

RMSE is calculated as:

$$
RMSE=\sqrt{MSE}
$$

RMSE is expressed in the same units as the target variable, making it easier to interpret the typical prediction error.

---

## ⚙️ Data Preprocessing

Before training the model, the input features go through several preprocessing steps.

### Numerical Feature Selection

Only numerical columns are selected from the dataset.

```python
X = df.select_dtypes(include=[np.number]).copy()
```

The target variable, `exam_score`, is then removed from the feature matrix.

---

### Missing Values

Missing values are replaced with zero:

```python
X = X.fillna(0)
```

---

### Feature Standardization

The input features are standardized using:

$$
z=\frac{x-\mu}{\sigma}
$$

where:

- $x$ = original feature value
- $\mu$ = feature mean
- $\sigma$ = feature standard deviation

This helps gradient descent converge more effectively when different features have different scales.

---

### Intercept

An additional column containing ones is added to represent the intercept:

```python
X["intercept"] = 1
```

This allows the model to learn $\theta_0$.

---

## 🏋️ Training

The model starts with all parameters initialized to zero:

```python
parameters = np.zeros(X.shape[1])
```

Gradient descent is then performed repeatedly.

Example configuration:

```python
iterations = 10000
learningRate = 0.01
```

At every iteration:

1. Calculate predictions.
2. Calculate the gradient.
3. Update the parameters.
4. Repeat.

The core update is:

```python
parameters -= learningRate * gradient
```

---

## 📊 Visualization

The project generates an **Actual vs. Predicted Exam Scores** scatter plot.

The red dashed line represents:

$$
y=x
$$

This is the ideal prediction line.

If a prediction lies exactly on the line:

$$
\hat{y}=y
$$

meaning the predicted score is equal to the actual score.

The closer the points are to this line, the smaller the prediction error.

---

## 🛠️ Technologies Used

- **Python**
- **NumPy** — numerical computation and matrix operations
- **Pandas** — dataset loading and preprocessing
- **Matplotlib** — data visualization

No machine learning library such as Scikit-learn is used to train the regression model.

The regression algorithm, gradient calculation, and parameter updates are implemented manually.

---

## 📁 Project Structure

```text
.
├── student_exam_performance.csv
├── main.py
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

### 2. Install the dependencies

```bash
pip install numpy pandas matplotlib
```

### 3. Place the dataset

Make sure the following file is located in the project directory:

```text
student_exam_performance.csv
```

### 4. Run the program

```bash
python main.py
```

The program will train the linear regression model, calculate the prediction error, and display the actual-vs-predicted visualization.

---

## 📈 Evaluation

The model reports:

```text
MSE: <value>
RMSE: <value>
```

### MSE

Measures the average squared difference between predicted and actual scores.

### RMSE

Provides the typical prediction error in the same units as the exam score.

---

## 🎯 Learning Objectives

This project was built to understand the fundamentals behind linear regression and gradient descent.

Key concepts explored:

- Linear regression
- Multiple linear regression
- Matrix/vector notation
- Cost functions
- Partial derivatives
- Gradients
- Gradient descent
- Learning rate
- Model parameters
- Intercept/bias
- Feature standardization
- Mean Squared Error
- Root Mean Squared Error
- Vectorized NumPy operations
- Data visualization

---

## 🔬 Why From Scratch?

Instead of using:

```python
from sklearn.linear_model import LinearRegression
```

the model is implemented manually.

This makes it possible to understand what happens internally during training:

$$
X
\rightarrow
X\theta
\rightarrow
\text{error}
\rightarrow
\nabla J(\theta)
\rightarrow
\theta_{\text{new}}
$$

Rather than treating linear regression as a black box, this project focuses on the underlying mathematics and optimization process.

---

## 🔮 Future Improvements

Possible improvements include:

- [ ] Train/test dataset split
- [ ] Validation dataset
- [ ] Feature selection
- [ ] Outlier detection
- [ ] Loss vs. iteration visualization
- [ ] Learning-rate experiments
- [ ] Hyperparameter tuning
- [ ] Regularization
- [ ] Ridge regression
- [ ] Lasso regression
- [ ] Comparison with Scikit-learn
- [ ] Prediction for individual students
- [ ] Model performance comparison against a mean-prediction baseline

---

## 📚 Key Takeaway

This project demonstrates how a linear regression model can be built from the ground up using matrix operations and gradient descent.

The central idea is:

$$
\boxed{
\theta
\leftarrow
\theta
-
\frac{\alpha}{m}
X^T(X\theta-y)
}
$$

By repeatedly applying this update, the model learns parameters that minimize the prediction error.

---

## 👨‍💻 Author

Prinjal Poudel

Built as a from-scratch machine learning project to explore the mathematics, optimization, and implementation behind linear regression.
