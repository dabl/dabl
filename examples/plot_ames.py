"""
Ames Housing Dataset Visualization
====================================
"""
from dabl import plot_supervised
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_pickle("ames_housing.pkl.bz2")

plot_supervised(df, 'SalePrice')
plt.show()
