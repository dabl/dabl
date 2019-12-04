"""
Human Activity Recognition Visualization
==========================================
"""
# sphinx_gallery_thumbnail_number = 3
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from dabl import plot

X, y = fetch_openml('har', as_frame=True, return_X_y=True)

plot(X, y)
plt.show()
