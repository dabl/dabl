"""
Adult Census Dataset Visualization
====================================
"""
from dabl import plot
from dabl.datasets import load_adult
import matplotlib.pyplot as plt

# load the adult census housing dataset
# returns a plain dataframe
data = load_adult()

plot(data, 'income', scatter_alpha=.1)
plt.show()
