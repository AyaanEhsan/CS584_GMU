import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#reading train dataset
data = np.loadtxt('./TestData_P1.txt')

#updating the centroids based on Sum of squared distances
def updateCentroids(data, centroids, assignedClusters):
    centroids = []
    for cl in range(len(centroids)):
        centroids.append(np.mean([data[x] for x in range(len(data)) if assignedClusters[x] == cl], axis=0))
    return centroids

#Normalizing the data instead of using the raw data so that the data only ranges from 0 to 1
def normalizeData(data):
    data = (data- np.min(data))/ (np.max(data) - np.min(data))
    return data

#assigning clusters for the given instances
def assignClusters(data, centroids):
    cl = []
    for i in data:
        cl.append(np.argmin(np.sum((i.reshape((1, 2)) - centroids) ** 2, axis=1)))
    return cl

#calling the normalizeData function
ND_data = normalizeData(data)
#applying t-distributed stochastic neighbor embedding on the normalized data to reduce the dimensions
tsne = TSNE(n_components=2, verbose=1, perplexity=15, n_iter=300)
ND_data = tsne.fit_transform(ND_data)
#generating 3 random centroids from the normalized data
centroids = (np.random.normal(size=(3, 2)) * 0.0001) + np.mean(ND_data, axis=0).reshape((1, 2))
df_subset = pd.DataFrame()
#iterating 100 times and finding sum of squared distances to get accurate cluster division
for i in range(100):
    initialClusters = assignClusters(ND_data, centroids)
    centroids = updateCentroids(ND_data, centroids, initialClusters)
    centroids = np.array(centroids)
    if i==99:
        (pd.DataFrame(initialClusters).replace([0,1,2],[1,2,3])).to_csv('part1_output.txt',index=False,header=False)

#plotting the clustered data
plt.scatter(ND_data[:, 0], ND_data[:, 1])
plt.scatter(centroids[:, 0], centroids[:, 1],c='yellow',label='centroids')
plt.legend()
plt.show()
