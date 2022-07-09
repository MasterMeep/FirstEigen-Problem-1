import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import time
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import plotly.graph_objs as go


import pandas_helper_calc



df = pd.read_csv('anomaly.csv')

X = df.drop(['CustomerID', 'Customer_Type', 'CustomerTier', 'ProductTier', 'Customer_govt_id', 'Customer_Phone', 'ZipCode', 'SSN_Individual', 'QSDate', 'TransactionDate', 'ClosingDate', 'Country'], axis=1)
#X = df.drop(['Loan_ID', 'effective_date', 'due_date', 'paid_off_time', 'past_due_days', 'education', 'loan_status', 'Gender'], axis=1)
#X = X[:299]

min_samples = X.shape[1]*2

db = DBSCAN(eps=0.61, min_samples=min_samples).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)


print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

index = [i for i, x in enumerate(list(labels)) if x == -1]

print(df.iloc[[*index]])




from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=n_clusters_, random_state=0).fit(X)

#prediction = kmeans.predict(pd.DataFrame({0:X.iloc[:,0],1:X.iloc[:,1],2:X.iloc[:,2]}))
prediction = kmeans.predict(pd.DataFrame({0:X.iloc[:,0],1:X.iloc[:,1]}))

print(pd.concat([pd.DataFrame(prediction),pd.DataFrame(labels)]).drop_duplicates(keep=False))
print(set(prediction), len(prediction))

X_tsne = X

if X.shape[1] > 2:
    X_tsne = pd.DataFrame(TSNE(n_components=2, random_state=0).fit_transform(X))

plt.scatter(X_tsne.iloc[:,0], X_tsne.iloc[:,1], c=labels, cmap='rainbow')
plt.colorbar()
plt.title("DBSCAN")
plt.show()



neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)

distLen = len(distances)

distances = np.sort(distances, axis=0)
distances = distances[:,1]


plt.plot(distances)

plt.show()
