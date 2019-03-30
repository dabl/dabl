"""
Wine Classification Dataset Visualization
==========================================
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from dabl import plot_supervised
from dabl.utils import data_df_from_bunch

wine_bunch = load_wine()
wine_df = data_df_from_bunch(wine_bunch)

plot_supervised(wine_df, 'target')
plt.show()
