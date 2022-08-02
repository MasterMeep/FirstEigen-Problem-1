from sklearn.cluster import KMeans
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np

df = pd.read_csv('loans.csv')

X = numeric = df.select_dtypes([np.number]).dropna(axis='columns')

min_samples = X.shape[1]*2

db = DBSCAN(eps=3, min_samples=min_samples).fit(X)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

index = [i for i, x in enumerate(list(labels)) if x == -1]
print(X.iloc[[*index]])

print(set(labels))

labelIndicies = list(np.where(labels==-1)[0])
X_ = np.delete(np.array(X), labelIndicies, axis=0)
kmeans = KMeans(n_clusters=n_clusters_, random_state=0).fit(X_)

prediction = kmeans.predict([[8000, 7000, 34000]])

print(set(prediction))

#print(pd.concat([pd.DataFrame(prediction),pd.DataFrame(labels)]).drop_duplicates(keep=False))