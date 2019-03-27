"""
Successive Halving heatmap of scores
====================================
"""

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
import numpy as np
import pandas as pd

from fml import GridSuccessiveHalving


rng = np.random.RandomState(0)
X, y = datasets.make_classification(n_samples=1000, random_state=rng)

gammas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
Cs = [1, 10, 100, 1e3, 1e4, 1e5]
param_grid = {'gamma': gammas, 'C': Cs}

clf = SVC(random_state=rng)
rsh = GridSuccessiveHalving(
    estimator=clf,
    param_grid=param_grid,
    budget_on='n_samples',  # budget is the number of samples
    max_budget='auto',  # max_budget=n_samples
    force_exhaust_budget=True,
    cv=5,
    ratio=2,
    random_state=rng)
rsh.fit(X, y)

results = pd.DataFrame.from_dict(rsh.cv_results_)
results['params_str'] = results.params.apply(str)
iterations = results.groupby(['param_gamma', 'param_C']).iter.max()
iterations_matrix = iterations.values.reshape(len(gammas), len(Cs))
scores = results.groupby(['param_gamma', 'param_C']).mean_test_score.max()
scores_matrix = scores.values.reshape(len(gammas), len(Cs))

fig, ax = plt.subplots()
im = ax.imshow(scores_matrix)

cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('max mean_test_score', rotation=-90, va="bottom")

ax.set_xticks(np.arange(len(Cs)))
ax.set_xticklabels(['{:.0E}'.format(x) for x in Cs])
ax.set_xlabel('C')

ax.set_yticks(np.arange(len(gammas)))
ax.set_yticklabels(['{:.0E}'.format(x) for x in gammas])
ax.set_ylabel('gamma')

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(gammas)):
    for j in range(len(Cs)):
        text = ax.text(j, i, iterations_matrix[i, j],
                       ha="center", va="center", color="w", fontsize=20)

ax.set_title("Highest score and highest reached iteration for each "
             "parameter combination")
fig.tight_layout()
plt.show()
