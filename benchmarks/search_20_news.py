"""
Benchmarking successive halving on a text processing pipeline
=============================================================


                    training time   test score   best CV score
---------------------------------------------------------------
GridSearchCV             19984.9 s      0.8567          0.9262
GridSuccessiveHalving      598.4 s      0.8514          0.8811
---------------------------------------------------------------

Best Params GridSuccessiveHalving
{'clf__C': 1000.0, 'vect': TfidfVectorizer(), 'vect__ngram_range': (1, 1)}

Best Params GridSearchCV
{'clf__C': 1000.0,
  'vect': TfidfVectorizer(ngram_range=(1, 2)),
  'vect__ngram_range': (1, 2)}
"""

import numpy as np

from time import time
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from dabl.search import GridSuccessiveHalving

data_train = fetch_20newsgroups(subset="train")
data_test = fetch_20newsgroups(subset="test")

pipe = Pipeline([('vect', CountVectorizer()), ('clf', LogisticRegression())])
param_grid = {
    'vect': [TfidfVectorizer(), CountVectorizer()],
    'clf__C': np.logspace(-3, 3, 7),
    'vect__ngram_range': [(1, 1), (1, 2), (1, 3)]}
print("Parameter grid:")
print(param_grid)

sh = GridSuccessiveHalving(pipe, param_grid, cv=5)
print("Start successive halving")
tick = time()
sh.fit(data_train.data, data_train.target)
print("Training Time Successive Halving", time() - tick)
print("Test Score Successive Halving: ",
      sh.score(data_test.data, data_test.target))
print("Parameters Successive Halving: ", sh.best_params_)

gs = GridSearchCV(pipe, param_grid, cv=5)
print("Start Grid Search")
tick = time()
gs.fit(data_train.data, data_train.target)
print("Training Time Grid Search: ", time() - tick)
print("Test Score Grid Search: ", gs.score(data_test.data, data_test.target))
print("Parameters Grid Search: ", gs.best_params_)
