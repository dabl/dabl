"""
Ames Housing Dataset Visualization
====================================
"""
from dabl import plot_supervised
from dabl.datasets import load_adult
import matplotlib.pyplot as plt

# load the adult census housing dataset
# returns a plain dataframe
data = load_adult()

plot_supervised(data, 'income',
                type_hints={'age': 'continuous',
                            'hours-per-week': 'continuous',
                            'capital-gain': 'continuous'},
                scatter_alpha=.1)
plt.show()
