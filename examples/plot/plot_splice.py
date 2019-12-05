"""
Splice dataset visualization
==========================================
A nice illustration of the mosaic plot.
"""
# sphinx_gallery_thumbnail_number = 2
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from dabl import plot

X, y = fetch_openml('splice', as_frame=True, return_X_y=True)

plot(X, y)
plt.show()
