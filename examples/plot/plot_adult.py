"""
Adult Census Dataset Visualization
====================================
"""
# sphinx_gallery_thumbnail_number = 2
from dabl import plot
from dabl.datasets import load_adult
import matplotlib.pyplot as plt

# load the adult census dataset
# returns a plain dataframe
data = load_adult()
plot(data, target_col='income', scatter_alpha=.1)
plt.show()
