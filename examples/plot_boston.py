"""
Boston Housing Dataset Visualization
====================================
"""
from sklearn.datasets import load_boston
from dabl import plot_supervised
import matplotlib.pyplot as plt
import pandas as pd

boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['MEDV'] = boston.target

plot_supervised(df, 'MEDV')
plt.show()
