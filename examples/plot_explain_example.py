"""
Model Explanation
=================
"""
from dabl.models import SimpleClassifier
from dabl.explain import explain
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()

X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target)

sc = SimpleClassifier()

sc.fit(X_train, y_train)

explain(sc, X_test, y_test)
