"""
Adult Census Dataset Visualization
====================================
"""
from dabl import plot_supervised
from dabl.datasets import load_adult
import matplotlib.pyplot as plt

# load the adult census housing dataset
# returns a plain dataframe
data = load_adult()

plot_supervised(data, 'income', scatter_alpha=.1)
plt.show()
