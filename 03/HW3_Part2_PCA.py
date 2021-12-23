import pandas as pd
import numpy as np
from sklearn.metrics import  silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import sklearn.metrics as metrics
import warnings

warnings.filterwarnings('ignore')
#Loading the data
data = pd.read_csv('./TestData_P2.txt', header= None)

#Checking if there are any null values in the data
nullValuesCount=data.isnull().sum()

#Removing the colums which contain only zeros
preProcessedData = data.loc[:, (data != 0).any(axis=0)]

#function used to Normalize the data instead of using the raw data so that the data only ranges from 0 to 1
def normalizeData(data):
    data = (data- np.min(data))/ (np.max(data) - np.min(data))
    return data

#normalizing the data
ND_data = normalizeData(preProcessedData.to_numpy(copy= True))

#applying Principal component analysis on the normalized data to reduce the dimensions
pca = PCA(2)
reduced_ND_data = pca.fit_transform(ND_data)

#assigning clusters for the given instances
def assignClusters(data, centroids):
  cl = []
  for i in data:
    cl.append(np.argmin(np.sum((i.reshape((1, 2)) - centroids) ** 2, axis=1)))
  return cl

#updating the centroids based on Sum of squared distances
def updateCentroids(centroids, cluster_assignments, data):
  previous_centroids = centroids.copy()
  centroids = []
  for cl in range(len(cluster_assignments)):
    centroids.append(np.mean([data[x] for x in range(len(data)) if cluster_assignments[x] == cl], axis=0))
  return previous_centroids,previous_centroids

#function that repeats the steps to move the centroid
def iterations(reduced_ND_data, k):  
  initialCentroids = reduced_ND_data[np.random.choice(reduced_ND_data.shape[0], size = k, replace= False)]
  sameCentroidsFlag = False
  current_centroids = np.empty(initialCentroids.shape)
  previous_centroids = initialCentroids
  initiallyAssignedClusters = assignClusters(reduced_ND_data, initialCentroids)
  newClusters = np.empty(len(initiallyAssignedClusters))
  max_iterations = 100
  score = 0.0
  while max_iterations > 0 and (not(sameCentroidsFlag)):
    previous_centroids, current_centroids = updateCentroids(initialCentroids,initiallyAssignedClusters, reduced_ND_data)
    sameCentroidsFlag = (np.array_equal(previous_centroids,current_centroids) and score > 0.035)
    newClusters=assignClusters(reduced_ND_data, current_centroids)
    score = silhouette_score(reduced_ND_data, newClusters)
    max_iterations -=1
  return newClusters

#clusters visualization function
def visualize(reduced_ND_data, newClusters):
  df = pd.DataFrame()
  df['X'] = reduced_ND_data[:,0]
  df['Y'] = reduced_ND_data[:,1]
  df['clusters'] = newClusters
  plt.figure(figsize=(10,8))
  sns.scatterplot(
      x="X", y="Y",
      hue= df['clusters'],
      palette=sns.color_palette('husl', as_cmap=True),
      data=df,
      legend="full",
      alpha=0.3
  )
  plt.title("Clustering of Image Dataset using PCA")
  plt.show()

kVlaues=[]
silhouette_scores=[]
#plot for K value vs Silhouette score
for i in range(2,22,2):
    labels=iterations(reduced_ND_data, k=i)
    silScore=metrics.silhouette_score(reduced_ND_data,labels,metric="euclidean",sample_size=1000,random_state=200)
    print ("Silhouette score for "+str(i)+" clusters = "
           +str(silScore))
    kVlaues.append(i)
    silhouette_scores.append(silScore)

plt.plot(kVlaues, silhouette_scores)
plt.xlabel("K Value")
plt.ylabel("silhouette score")
plt.title("K Value vs Silhouette Scores")
plt.show()

newClusters = iterations(reduced_ND_data, k=10)
(pd.DataFrame(newClusters).replace([0,1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9,10])).to_csv('part2_output_PCA.txt',index=False,header=False)
#calling the visualization function
visualize(reduced_ND_data, newClusters)