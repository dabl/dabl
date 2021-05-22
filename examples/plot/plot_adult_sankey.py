"""
Adult Census Dataset Visualization
====================================
"""
from dabl.plot import plot_sankey
from dabl.plot import plot_classification_categorical
from dabl.datasets import load_adult
import matplotlib.pyplot as plt

# load the adult census dataset
# returns a plain dataframe
data = load_adult()
# visualize the joint distribution of the 5 most important categorical variables
plot_sankey(data, 'income', figure=plt.figure(figsize=(12, 4), dpi=150))
# compare to mosaic plots which show only univariate effects
plot_classification_categorical(data, 'income')
plt.show()
