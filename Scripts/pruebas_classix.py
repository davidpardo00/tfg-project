# import classix; 
# classix.cython_is_available()
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from classix import CLASSIX

X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=1)
clx = CLASSIX(radius=0.5, minPts=13)
clx.fit(X)

plt.figure(figsize=(10,10))
plt.scatter(X[:,0], X[:,1], c=clx.labels_)
plt.show()