import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



data = pd.read_csv("KNNAlgorithmDataset.csv")

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

dist = []
y_pred = []
for point1 in x_test:
    for point2 in x_train:
        dist.append(euclideanDistance(point1,point2))

    points = np.argsort(dist)[:k]
    pointClass = [y[i] for i in points]
    count1 = pointClass.count(1)
    count2 = pointClass.count(2)

    if count1 > count2 :
        y_pred.append(1)
    else:
        y_pred.append(2)


print(y_pred)