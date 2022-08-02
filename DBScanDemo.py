import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import numpy as np

df = pd.read_csv('anomaly.csv')

X = numeric = df.select_dtypes([np.number]).dropna(axis='columns')

print(X)

min_samples = X.shape[1]*2


db = DBSCAN(eps=3, min_samples=min_samples).fit(X)
labels = db.labels_

if X.shape[1] > 2:
    X = pd.DataFrame(TSNE(n_components=2, random_state=0).fit_transform(X))

plt.scatter(X.iloc[:,0], X.iloc[:,1], c=labels, cmap='rainbow')
plt.colorbar()
plt.title("DBSCAN")
plt.axis('off')
plt.show()

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)



print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

index = [i for i, x in enumerate(list(labels)) if x == -1]

print(df.iloc[[*index]])