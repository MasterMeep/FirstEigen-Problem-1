import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


df = pd.read_csv('loans.csv')
X = numeric = df.select_dtypes([np.number]).dropna(axis='columns')

neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)

distLen = len(distances)

distances = np.sort(distances, axis=0)
distances = distances[:,1]

threesigma = distances.mean() + 2 * distances.std()


print(distances[np.abs(distances-threesigma).argmin()-1])


plt.plot(distances)
plt.axvline(np.abs(distances-threesigma).argmin()-1, color="red")
plt.show()