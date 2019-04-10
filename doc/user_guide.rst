.. title:: User guide : contents

.. _user_guide:

==================================================
Machine Learning with dabl
==================================================

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

dabl is meant to support you in the following tasks, in order:

Data cleaning
-------------
>>> import dabl
>>> import pandas as pd
>>> data = pd.read_csv(dabl.datasets.data_path("adult.csv.gz"))
>>> data_clean = dabl.clean(data)[::10]

The first step in any data analysis is data cleaning. dabl tries to detect the
types of your data and apply appropriate conversions.  It also tries to detect
potential data quality issues.
The field of data cleaning is impossibly broad, and dabl's approaches are by no
means sophisticated.  The goal of dabl is to get the data "clean enough" to
create useful visualizations and models, and to allow the user to perform
custom cleaning operations themselves.
In particular if the detection of semantic types (continuous, categorical,
ordinal, text, etc) fails, the user can provide ``type_hints``:

>>> data_clean = dabl.clean(data, type_hints={"capital-gain": "continuous"})

Exploratory Data analysis
-------------------------
>>> dabl.plot_supervised(data, target_col="income")
Target looks like classification


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
>>> ec = dabl.SimpleClassifier(random_state=0).fit(data, target_col="income")
DummyClassifier(strategy='prior')
accuracy: 0.759    average_precision: 0.241    recall_macro: 0.500    roc_auc: 0.500
new best (using recall_macro):
accuracy             0.759
average_precision    0.241
recall_macro         0.500
roc_auc              0.500
Name: DummyClassifier(strategy='prior'), dtype: float64
GaussianNB()
accuracy: 0.407    average_precision: 0.288    recall_macro: 0.605    roc_auc: 0.607
new best (using recall_macro):
accuracy             0.407
average_precision    0.288
recall_macro         0.605
roc_auc              0.607
Name: GaussianNB(), dtype: float64
MultinomialNB()
accuracy: 0.831    average_precision: 0.773    recall_macro: 0.815    roc_auc: 0.908
new best (using recall_macro):
accuracy             0.831
average_precision    0.773
recall_macro         0.815
roc_auc              0.908
Name: MultinomialNB(), dtype: float64
DecisionTreeClassifier(class_weight='balanced', max_depth=1)
accuracy: 0.710    average_precision: 0.417    recall_macro: 0.759    roc_auc: 0.759
DecisionTreeClassifier(class_weight='balanced', max_depth=5)
accuracy: 0.784    average_precision: 0.711    recall_macro: 0.811    roc_auc: 0.894
DecisionTreeClassifier(class_weight='balanced', min_impurity_decrease=0.01)
accuracy: 0.718    average_precision: 0.561    recall_macro: 0.779    roc_auc: 0.848
LogisticRegression(C=0.1, class_weight='balanced', solver='lbfgs')
accuracy: 0.819    average_precision: 0.789    recall_macro: 0.832    roc_auc: 0.915
new best (using recall_macro):
accuracy             0.819
average_precision    0.789
recall_macro         0.832
roc_auc              0.915
Name: LogisticRegression(C=0.1, class_weight='balanced', solver='lbfgs'), dtype: float64
Best model:
LogisticRegression(C=0.1, class_weight='balanced', solver='lbfgs')
Best Scores:
accuracy             0.819
average_precision    0.789
recall_macro         0.832
roc_auc              0.915
Name: LogisticRegression(C=0.1, class_weight='balanced', solver='lbfgs'), dtype: float64



Fit an initial model. The SimpleClassifier first tries several baseline and
instantaneous models, potentially on subsampled data, to get an idea of what a
low baseline should be.
This again is a good place to surface data leakage, as well as find the main
discriminative features in the dataset.  The ``SimpleClassifier`` allows
specifying data in the scikit-learn-style ``fit(X, y)`` with a 1d y and
features ``X``, or with ``X`` being a dataframe, and by specifying the target
column inside of X as``target_col``.

The SimpleClassifier also performs preprocessing such as missing value
imputation and one-hot-encoding.  You can inspect the model using:

>>> dabl.explain(ec)

This can lead to additional insights and guide costom processing and
cleaning of the data.

Enhanced Model Building
------------------------
>>> # ac = AnyClassifier().fit(data, target_col="income") not implemented yet

After creating an initial model, it's interesting to explore more powerful
models such as tree ensembles.  ``AnyClassifier`` searches over a space of
models that commonly perform well, and identifies promising candidates.  If
your goal is prediction, ``AnyClassifier`` can provide a strong baseline for
further investigation.  Again, we can inspect our model to understand it
better:

>>> # explain(ac)


Explainable Model Building
---------------------------
TODO this is not done yet!

Sometimes, explainability of a model can be more important than performance. A
complex model can serve as a good benchmark on what is achievable on a certain
dataset. After this benchmark is established, it is interesting to see if we
can build a model that is interpretable while still providing competitive
performance.

>>> # xc = ExplainableClassifier().fit(data, target_col="income")


.. include:: ./successive_halving.rst


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
