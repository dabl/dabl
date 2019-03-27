.. title:: User guide : contents

.. _user_guide:

==================================================
Friendly Machine Learning
==================================================

Estimator
---------

The central piece of transformer, regressor, and classifier is
:class:`sklearn.base.BaseEstimator`. All estimators in scikit-learn are derived
from this class. In more details, this base class enables to set and get
parameters of the estimator. It can be imported as::

    >>> from sklearn.base import BaseEstimator

Once imported, you can create a class which inherate from this base class::

    >>> class MyOwnEstimator(BaseEstimator):
    ...     pass

