.. _successive_halving_user_guide:

Searching optimal parameters with successive halving
----------------------------------------------------

``dabl`` provides the :class:`dabl.search.GridSuccessiveHalving` and
:class:`dabl.search.RandomSuccessiveHalving` estimators that can be used to
search a parameter space using successive halving [1]_ [2]_. Successive
halving is an iterative selection process where all candidates are evaluated
with a small amount of resources at the first iteration. Only a subset of
these candidates are selected for the next iteration, which will be
allocated more resources. What defines a resource is typically the number of
samples to train on, or the number of trees for a gradient boosting /
decision forest estimator.

As illustrated in the figure below, only a small subset of candidates 'survive'
until the last iteration. These are the candidates that have consistently been
part of the best candidates across all iterations.

#FIXME: Put figure from `plot_successive_halving_iterations.py` here

The amount of resources ``r_i`` allocated for each candidate at iteration
``i`` is controlled by the parameters ``ratio`` and ``r_min`` as follows::

    r_i = ratio**i * r_min

``r_min`` is the amount of resources used at the first iteration and
``ratio`` defines the proportions of candidates that will be selected for
the next iteration::

    n_candidates_to_keep = n_candidates_at_i // ratio

Note that each ``r_i`` is a multiple of both ``ratio`` and ``r_min``.

Choosing the budget
^^^^^^^^^^^^^^^^^^^

By default, the budget is defined as the number of samples. That is, each
iteration will use an increasing amount of samples to train on. You can however
manually specify a parameter to use as the budget with the ``budget_on``
parameter. Here is an example where the budget is defined as the number of
iterations of a random forest::

    >>> from sklearn.datasets import make_classification
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> import pandas as pd
    >>> from dabl.search import GridSuccessiveHalving
    >>>
    >>> parameters = {'max_depth': [3, 5, 10],
    ...               'min_samples_split': [2, 5, 10]}
    >>> base_estimator = RandomForestClassifier(random_state=0)
    >>> X, y = make_classification(n_samples=1000, random_state=0)
    >>> sh = GridSuccessiveHalving(base_estimator, parameters, cv=5,
    ...                            ratio=2,
    ...                            budget_on='n_estimators',
    ...                            max_budget=30,
    ...                            random_state=0,
    ...                            ).fit(X, y)
    >>> sh.best_estimator_
    RandomForestClassifier(...)

Note that it is not possible to budget on a parameter that is part of the
parameter space.

Exhausting the budget
^^^^^^^^^^^^^^^^^^^^^

As mentioned above, the first iteration uses ``r_min`` resources. If you have
a big budget, this may be a waste of resource::

    >>> from sklearn.datasets import make_classification
    >>> from sklearn.svm import SVC
    >>> import pandas as pd
    >>> from dabl.search import GridSuccessiveHalving
    >>> parameters = {'kernel': ('linear', 'rbf'),
    ...               'C': [1, 10, 100]}
    >>> base_estimator = SVC(gamma='scale')
    >>> X, y = make_classification(n_samples=1000)
    >>> sh = GridSuccessiveHalving(base_estimator, parameters, cv=5,
    ...                            ratio=2).fit(X, y)
    >>> results = pd.DataFrame.from_dict(sh.cv_results_)
    >>> results.groupby('iter').r_i.unique()
    iter
    0    [20]
    1    [40]
    2    [80]
    Name: r_i, dtype: object

The search process will only use 80 resources at most, while our maximum budget
is ``n_samples=1000``. Note in this case that ``r_min = r_0 = 20``. In order
for the last iteration to use as many resources as possible, you can use the
``force_exhaust_budget`` parameter::

    >>> sh = GridSuccessiveHalving(base_estimator, parameters, cv=5,
    ...                            ratio=2, force_exhaust_budget=True,
    ...                            ).fit(X, y)
    >>> results = pd.DataFrame.from_dict(sh.cv_results_)
    >>> results.groupby('iter').r_i.unique()
    iter
    0     [250]
    1     [500]
    2    [1000]
    Name: r_i, dtype: object


Since ``force_exhaust_budget`` chooses an appropriate ``r_min`` to start
with, ``r_min`` must be set to 'auto'.

Aggressive elimination of candidates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ideally, we want the last iteration to evaluate ``ratio`` candidates. We then
just have to pick the best one. When the number budget is small with respect to
the number of candidates, the last iteration may have to evaluate more than
``ratio`` candidates::

    >>> from sklearn.datasets import make_classification
    >>> from sklearn.svm import SVC
    >>> import pandas as pd
    >>> from dabl.search import GridSuccessiveHalving
    >>>
    >>>
    >>> parameters = {'kernel': ('linear', 'rbf'),
    ...               'C': [1, 10, 100]}
    >>> base_estimator = SVC(gamma='scale')
    >>> X, y = make_classification(n_samples=1000)
    >>> sh = GridSuccessiveHalving(base_estimator, parameters, cv=5,
    ...                            ratio=2,
    ...                            max_budget=40,
    ...                            aggressive_elimination=False,
    ...                            ).fit(X, y)
    >>> results = pd.DataFrame.from_dict(sh.cv_results_)
    >>> results.groupby('iter').r_i.unique()
    iter
    0    [20]
    1    [40]
    Name: r_i, dtype: object
    >>> results.groupby('iter').r_i.count()  # number of candidates used at each iteration
    iter
    0    6
    1    3
    Name: r_i, dtype: int64

Since we cannot use more than ``max_budget=40`` resources, the process has to
stop at the second iteration which evaluates more than ``ratio=2`` candidates.

Using the ``aggressive_elimination`` parameter, you can force the search
process to end up with less than ``ratio`` candidates at the last
iteration. To do this, the process will eliminate as many candidates as
necessary using ``r_min`` resources::

    >>> sh = GridSuccessiveHalving(base_estimator, parameters, cv=5,
    ...                            ratio=2,
    ...                            max_budget=40,
    ...                            aggressive_elimination=True,
    ...                            ).fit(X, y)
    >>> results = pd.DataFrame.from_dict(sh.cv_results_)
    >>> results.groupby('iter').r_i.unique()
    iter
    0    [20]
    1    [20]
    2    [40]
    Name: r_i, dtype: object
    >>> results.groupby('iter').r_i.count()  # number of candidates used at each iteration
    iter
    0    6
    1    3
    2    2
    Name: r_i, dtype: int64

Notice that we end with 2 candidates at the last iteration since we have
eliminated enough candidates during the first iterations, using ``r_i = r_min =
20``.


.. topic:: References:

    .. [1] K. Jamieson, A. Talwalkar,
       `Non-stochastic Best Arm Identification and Hyperparameter
       Optimization <http://proceedings.mlr.press/v51/jamieson16.html>`_, in
       proc. of Machine Learning Research, 2016.
    .. [2] L. Li, K. Jamieson, G. DeSalvo, A. Rostamizadeh, .A Talwalkar,
       `Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization
       <https://arxiv.org/abs/1603.06560>`_, in Machine Learning Research
       18, 2018.
