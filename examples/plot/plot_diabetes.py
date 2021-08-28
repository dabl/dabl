"""
Diabetes Dataset Visualization
==========================================
"""
               
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import plot_partial_dependence
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import datasets

diabetes = datasets.load_diabetes

X= pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y=pd.DataFrame(diabetes.target)

tree = DecisionTreeRegressor()
tree.fit(X, y)
tree_disp = plot_partial_dependence(tree, X, [0,1,2,3,4,5,6,7,8,9])
#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X, y)
#features = [(0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (2,3), (2,4), (2,5), (2,6), (2,7), (3,4), (3,5), (3,6), (3,7), (4,5), (4,6), (4,7), (5,6), (5,7), (6,7) ]
#features = [0,1]
#plot_partial_dependence(clf, X, features) 
#plt.tight_layout()
plt.show()