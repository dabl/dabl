"""
Scores of parameter combinations at each iteration of Successive Halving
========================================================================
"""
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
import numpy as np

from dabl import RandomSuccessiveHalving


rng = np.random.RandomState(0)

X, y = datasets.make_classification(n_samples=700, random_state=rng)

clf = RandomForestClassifier(n_estimators=20, random_state=rng)

param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 11),
              "min_samples_split": randint(2, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

rsh = RandomSuccessiveHalving(
    estimator=clf,
    param_distributions=param_dist,
    budget_on='n_samples',  # budget is the number of samples
    max_budget='auto',  # max_budget=n_samples
    n_candidates='auto',  # choose n_cdts so that last iter exhausts budget
    cv=5,
    ratio=2,
    random_state=rng)
rsh.fit(X, y)

results = pd.DataFrame(rsh.cv_results_)
results['params_str'] = results.params.apply(str)
mean_scores = results.pivot(index='iter', columns='params_str',
                            values='mean_test_score')
ax = mean_scores.plot(legend=False, alpha=.6)

labels = ['{}\nn_samples={}'.format(i, rsh._r_i_list[i])
          for i in range(rsh.n_iterations_)]
ax.set_xticklabels(labels)
ax.set_title('Candidate scores over iterations')
ax.set_ylabel('score')
plt.show()
