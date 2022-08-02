import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

df = pd.read_csv('loans.csv')

X = numeric = df.select_dtypes([np.number]).dropna(axis='columns')

if X.shape[1] > 2:
    X = pd.DataFrame(TSNE(n_components=2, random_state=0).fit_transform(X))

plt.scatter(X.iloc[:,0], X.iloc[:,1])
plt.colorbar()
plt.title("DBSCAN")
plt.axis('off')
plt.show()