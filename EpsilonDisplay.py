import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors



df = pd.read_csv('loans.csv')
X = numeric = df.select_dtypes([np.number]).dropna(axis='columns')


neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)

distLen = len(distances)

distances = np.sort(distances, axis=0)
distances = distances[:,1]


plt.plot(distances)

plt.show()