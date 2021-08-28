"""
Diabetes Dataset Visualization
==========================================
"""
import pandas as pd
from sklearn import datasets
from sklearn.inspection import plot_partial_dependence
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

diabetes= datasets.load_diabetes()

X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

enable_hist_gradient_boosting
est= HistGradientBoostingRegressor()
est.fit(X, y)

for i in range(0, 10):
  display= plot_partial_dependence(est, X, [i])

for i in range(0, 10):
  for j in range(0, 10):
    if(i < j):
      display= plot_partial_dependence(est, X, [(i, j)])
