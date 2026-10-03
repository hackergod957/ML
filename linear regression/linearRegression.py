import numpy as np
import pandas as pd
import matplotlib.pyplot as plt    

df = pd.read_csv("student_exam_performance.csv")


X = df.select_dtypes(include=[np.number]).copy()


if 'exam_score' in X.columns:
    X = X.drop(columns=['exam_score'])

X = X.fillna(0)
X = (X - X.mean()) / X.std()

y = df['exam_score']
X["dummmy"] = np.ones(X.shape[0])   

def gradientLossFunction(parameters,X,y):

    loss = 1/(X.shape[0]) * np.dot( np.transpose(X.to_numpy()) , (np.dot(X.to_numpy(),parameters).flatten() - y.to_numpy()))
    return loss 

parameters = np.zeros(X.shape[1])
iterations = 10000
learningRate = 0.01
for i in range(iterations):
    print(i)
    parameters -= learningRate * gradientLossFunction(parameters,X,y)


mse = 1/(X.shape[0]) * np.dot(np.dot(X.to_numpy(),parameters)-y.to_numpy(),np.dot(X.to_numpy(),parameters) - y.to_numpy() )
y_pred = np.dot(X.to_numpy(), parameters)

plt.scatter(y, y_pred, color='blue', alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', lw=2)
plt.title("Actual vs. Predicted Exam Scores")
plt.ylabel('Predicted Score')
plt.xlabel('Actual Score')
plt.show()

