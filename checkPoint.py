import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
from sklearn.cluster import DBSCAN


df = pd.read_csv('anomaly.csv')

points = numeric = df.select_dtypes([np.number]).dropna(axis='columns')

min_samples = points.shape[1]*2

db = DBSCAN(eps=0.608, min_samples=min_samples).fit(points)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

points['labels'] = labels.tolist()

clusters = dict(tuple(points.groupby('labels')))

import os
import sys
from descartes import PolygonPatch
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.getcwd()))
import alphashape

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from math import sqrt

def closest_node_dist(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    point = nodes[int(np.argmin(dist_2))]
    distance = sqrt((node[0]-point[0])**2+(node[1]-point[1])**2)
    return distance

def getHull(pts):
    alpha_shape = alphashape.alphashape(pts, 3)
    coords = list(alpha_shape.exterior.coords)
    return(coords)

def checkPoint(point, points, eps):
    hull = getHull(points)
    polygon = Polygon(hull)

    pt = Point(*point)
    val = pt.within(polygon)
    if val:
        return(True)
    else:
        if closest_node_dist(point, hull) <= eps:
            return(True)
    return(False)
            
eps = 0.608

pts = clusters[1].drop('labels', axis=1)

print((checkPoint((522, 51), pts, eps)))

plt.plot(*zip(*getHull(pts)))
plt.scatter(522, 51, c='red')
circle = plt.Circle((522, 51), eps, color='blue', fill=False)

ax = plt.gca()
ax.add_patch(circle)

plt.show()