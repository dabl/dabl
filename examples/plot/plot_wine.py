"""
Wine Classification Dataset Visualization
==========================================
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from dabl import plot
from dabl.utils import data_df_from_bunch

wine_bunch = load_wine()
wine_df = data_df_from_bunch(wine_bunch)

plot(wine_df, 'target')
plt.show()
