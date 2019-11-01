"""
Ames Housing Dataset Visualization
====================================
"""
from dabl import plot
from dabl.datasets import load_ames
import matplotlib.pyplot as plt

# load the ames housing dataset
# returns a plain dataframe
data = load_ames()

plot(data, 'SalePrice')
plt.show()
