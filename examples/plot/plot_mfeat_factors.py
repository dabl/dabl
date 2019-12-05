"""
mfeat-factors dataset visualization
==========================================
A multiclass dataset with 10 classes.
Linear discriminant analysis works surprisingly well!
"""
# sphinx_gallery_thumbnail_number = 5
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from dabl import plot

X, y = fetch_openml('mfeat-factors', as_frame=True, return_X_y=True)

plot(X, y)
plt.show()
