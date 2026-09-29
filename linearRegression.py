import numpy as np
import pandas as pd


df = pd.read_csv("student_exam_performance.csv")


X = df.drop(columns=['student_id', 'exam_score', 'pass_status', 'performance_grade', 'performance_level'])
y = df['exam_score']

def lossFunction(parameters,X,y):

    loss = 1/X.shape(1) * np.dot( np.transpose(X) , (np.dot(X,parameters) - y))
    return loss 