"""
Hyperparameter Tuning for SVM on handwritten digits
===================================================

On digits data:
Start successive halving
Training Time Successive Halving 3.1159074306488037
Test Score Successive Halving:  0.9911111111111112
Parameters Successive Halving:  {'C': 100.0, 'gamma': 0.1}
Start Grid Search
Training Time Grid Search:  39.42753505706787
Test Score Grid Search:  0.9911111111111112
Parameters Grid Search:  {'C': 10.0, 'gamma': 0.1}


"""
from time import time
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import fetch_openml, load_digits

from dabl.search import GridSuccessiveHalving

use_mnist = False

if use_mnist:
    digits = fetch_openml("mnist_784")
    scale = 255
else:
    digits = load_digits()
    scale = 16

X_train, X_test, y_train, y_test = train_test_split(
    digits.data / scale, digits.target, stratify=digits.target,
    random_state=42)

param_grid = {'C': np.logspace(-3, 2, 6), 'gamma': np.logspace(-3, 2, 6)}
gs = GridSearchCV(SVC(), param_grid, cv=5)


print("Parameter grid:")
print(param_grid)

sh = GridSuccessiveHalving(SVC(), param_grid, cv=5)
print("Start successive halving")
tick = time()
sh.fit(X_train, y_train)
print("Training Time Successive Halving", time() - tick)
print("Test Score Successive Halving: ",
      sh.score(X_test, y_test))
print("Parameters Successive Halving: ", sh.best_params_)

gs = GridSearchCV(SVC(), param_grid, cv=5)
print("Start Grid Search")
tick = time()
gs.fit(X_train, y_train)
print("Training Time Grid Search: ", time() - tick)
print("Test Score Grid Search: ", gs.score(X_test, y_test))
print("Parameters Grid Search: ", gs.best_params_)
