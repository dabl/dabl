"""
Ames Housing Dataset Visualization
====================================
"""
# sphinx_gallery_thumbnail_number = 3
from dabl import plot
from dabl.datasets import load_ames
import matplotlib.pyplot as plt

# load the ames housing dataset
# returns a plain dataframe
data = load_ames()

plot(data, 'SalePrice')
plt.show()
