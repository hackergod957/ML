import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("cluster_data.csv").to_numpy()

k = 5 

random_indices = np.random.choice(data.shape[0], size=k, replace=False)
centroids = data[random_indices]

clusters = [{"center": centroids[i], "points": []} for i in range(k)]
    

def euclideanDistance(center,point2) :
    dist = np.sqrt(np.sum( ( np.array(center) - np.array(point2) ) **2 ))
    return dist


def assignPoint(data,clusters):

    for c in clusters:

        c["points"] = []

    for point in data:

        dist = []
    
        for cluster in clusters:

            dist.append(euclideanDistance(cluster["center"],point))

        center = np.argmin(dist)
        clusters[center]["points"].append(point)
        

def calculateCentroids(Clusters):


    for cluster in Clusters:

        cluster["center"] = np.mean(cluster["points"], axis=0)  

    return clusters 
    
    


max_iters = 100

for _ in range(max_iters):
  old_centers = [c["center"].copy() for c in clusters]

  assignPoint(data, clusters)
  calculateCentroids(clusters)

  new_centers = [c["center"] for c in clusters]

  if np.allclose(old_centers, new_centers):
    break