.. title:: User guide : contents

.. _user_guide:

==================================================
Machine Learning with dabl
==================================================

dabl is meant to support you in the following tasks, in order:

Data cleaning
-------------
>>> data = pd.read_csv("adult.csv")
>>> data_clean = dabl.clean(data)

The first step in any data analysis is data cleaning. dabl tries to detect the
types of your data and apply appropriate conversions.  It also tries to detect
potential data quality issues.
The field of data cleaning is impossibly broad, and dabl's approaches are by no
means sophisticated.  The goal of dabl is to get the data "clean enough" to
create useful visualizations and models, and to allow the user to perform
custom cleaning operations themselves.
In particular if the detection of semantic types (continuous, categorical,
ordinal, text, etc) fails, the user can provide ``type_hints``:

>>> data_clean = dabl.clean(data, type_hints={"capital-gain": "continuous"}

Exploratory Data analysis
-------------------------
>>> dabl.plot_supervised(data, target_col="income")

The next step in any task should be exploratory data analysis. dabl provides a
high-level interface that summarizes several common high-level plots.  For low
dimensional datasets, all features are shown, for high dimensional datasets,
only the most informative features for the given task are shown.  This is
clearly not guaranteed to surface all interesting aspects with the data, or to
find all data quality issues.  However, it will give you a quick insight in to
what are the important features, their interactions, and how hard the problem
might be.  It also allows a good assessment of whether there is any data
leakage through spurious representations of the target in the data.

Initial Model Building
-----------------------
>>> ec = SimpleClassifier().fit(data, target_col="income")

Fit an initial model. The SimpleClassifier first tries several baseline and
instantaneous models, potentially on subsampled data, to get an idea of what a
low baseline should be.
This again is a good place to surface data leakage, as well as find the main
discriminative features in the dataset.  The ``SimpleClassifier`` allows
specifying data in the scikit-learn-style ``fit(X, y)`` with a 1d y and
features ``X``, or with ``X`` being a dataframe, and by specifying the target
column insided of X as``target_col``.

The SimpleClassifier also performs preprocessing such as missing value
imputation and one-hot-encoding.  You can inspect the model using:

>>> explain(ec)

This can lead to additional insights and guide costom processing and
cleaning of the data.

Enhanced Model Building
------------------------
>>> ac = AnyClassifier().fit(data, target_col="income")

After creating an initial model, it's interesting to explore more powerful
models such as tree ensembles.  ``AnyClassifier`` searches over a space of
models that commonly perform well, and identifies promising candidates.  If
your goal is prediction, ``AnyClassifier`` can provide a strong baseline for
further investigation.  Again, we can inspect our model to understand it
better:

>>> explain(ac)


Explainable Model Building
---------------------------
TODO this is not done yet!

Sometimes, explainability of a model can be more important than performance. A
complex model can serve as a good benchmark on what is achievable on a certain
dataset. After this benchmark is established, it is interesting to see if we
can build a model that is interpretable while still providing competitive
performance.

>>> xc = ExplainableClassifier().fit(data, target_col="income")

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
    >>>
    >>> parameters = {'max_depth': [3, 5, 10],
    >>>               'min_samples_split': [2, 5, 10]}
    >>> base_estimator = RandomForestClassifier()
    >>> X, y = make_classification(n_samples=1000)
    >>> sh = GridSuccessiveHalving(base_estimator, parameters, cv=5,
    ...                            ratio=2,
    ...                            budget_on='n_estimators',
    ...                            max_budget=30,
    ...                            ).fit(X, y)
    >>> sh.best_estimator_
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=5, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=10,
                           min_weight_fraction_leaf=0.0, n_estimators=8,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)

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
        0    [250]
        1    [500]
        2    [1000]
        Name: r_i, dtype: object


Since ``force_exhaust_budget`` chooses an appropriate ``r_min`` to start
with, ``r_min`` must be set to 'auto'.

Aggressive elimination of candidates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ideally, we want the last iteration to evaluate ``ratio`` candidates. We then
just have to pick the best one. When the number budget is small with respect to
the number of candidates, the last iteration may have to evaluate more than
``ratio`` candidates.::
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

Philosophy
----------
The idea behind dabl is to jump-start your supervised learning task.  dabl has
several tools that make it easy to clean and inspect your data, and create
strong baseline models.

Building machine learning models is an inherently iterative task with a human
in the loop.  Big jumps in performance are often achieved by better
understanding of the data and task, and more appropriate features.  dabl tries
to provide as much insight into the data as possible, and enable interactive
analysis.

Many analyses start with the same rote tasks of cleaning and basic data
visualization, and initial modelling.  dabl tries to make these steps as easy
as possible, so that you can spend your time thinking about the problem and
creating more intesting custom analyses.

There are two main packages that dabl takes inspiration from and that dabl
builds upon: scikit-learn and auto-sklearn.  But the design philosophies and
use-cases are quite different. Scikit-learn provides many essential building
blocks, but is build on the idea to exactly what the user asks for. That
requires specifying every step of the processing in detail.  dabl on the other
hand has a best-guess philosophy, tries to do something sensible, and then
provides tools for the user to inspect and evaluate the results to judge them.
auto-sklearn on the other hand is completely automatic and black-box. It
searches a vast space of models and constructs complex ensemles of high
accuracy, taking a substantial amount of computation and time in the process.
The goal of auto-sklearn is to build the best model possible given the data.

dabl on the other hand tries to enable quick iteration, and enable the user to
quickly iterate and get a grasp on the properties of the data at hand and the
fitted models.
>>>>>>> master



Limitations
-----------
Right now dabl does not deal with text data and time series data.  It also does
not consider neural network models.  Image, audio and video data is considered
out of scope.  All current implementation are quite rudimentary and rely
heavily on heuristics. The goal is to replace these with more principled
approaches where this provides a benefit.


Future Goals and Roadmap
-------------------------
dabl aims to provide easy-to-use, turn-key solutions for supervised machine
learning that strongly encourage iterative and interactive model building.
Key ingedients to achieve this are:

- Ready-made visualizations
- model diagnostics
- Efficient model search
- Type detection
- Automatic preprocessing
- portfolios of well-performing pipelines

The current version of dabl only provides very simple implementations of these,
but the goal is for dabl to contain more advanced solutions while providing a
simple user interface and strong anytime performance.
