"""
Benchmarking successive halving on a text processing pipeline
=============================================================


                    training time   test score   best CV score
---------------------------------------------------------------
RandomizedSearchCV
RandomSuccessiveHalving
---------------------------------------------------------------

Best Params RandomSuccessiveHalving

Best Params RandomizedSearchCV
"""

import numpy as np
import scipy as sp

from time import time
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

from dabl.search import RandomSuccessiveHalving


class log_uniform():
    def __init__(self, a=-1, b=0, base=10):
        self.loc = a
        self.scale = b - a
        self.base = base

    def rvs(self, size=1, random_state=None):
        uniform = sp.stats.uniform(loc=self.loc, scale=self.scale)
        return np.power(self.base, uniform.rvs(size=size,
                                               random_state=random_state))[0]


data_train = fetch_20newsgroups(subset="train")
data_test = fetch_20newsgroups(subset="test")

pipe = Pipeline([('vect', CountVectorizer()),
                 ('clf', LogisticRegression(tol=0.01, solver='saga',
                                            penalty='elasticnet'))])
param_dist = {
    'clf__C': log_uniform(-3, 3),
    'clf__l1_ratio': sp.stats.uniform(0, 1),
    'vect': [TfidfVectorizer(), CountVectorizer()],
    'vect__min_df': sp.stats.randint(0, 10),
    'vect__ngram_range': [(1, 1), (1, 2), (1, 3)]}
print("Parameter dist:")
print(param_dist)

sh = RandomSuccessiveHalving(pipe, param_dist, cv=5, verbose=10)
print("Start successive halving")
tick = time()
sh.fit(data_train.data, data_train.target)
print("Training Time Successive Halving", time() - tick)
print("Test Score Successive Halving: ",
      sh.score(data_test.data, data_test.target))
print("Parameters Successive Halving: ", sh.best_params_)

gs = RandomizedSearchCV(pipe, param_dist, cv=5, verbose=10)
print("Start Grid Search")
tick = time()
gs.fit(data_train.data, data_train.target)
print("Training Time Grid Search: ", time() - tick)
print("Test Score Grid Search: ", gs.score(data_test.data, data_test.target))
print("Parameters Grid Search: ", gs.best_params_)
