#############################
dabl API
#############################

This is a list of all functions and classes provided by dabl.

.. currentmodule:: dabl

High-level API
==============
.. autosummary::
   :toctree: generated/
   :template: function.rst

   clean
   detect_types
   explain
   plot

.. autosummary::
   :toctree: generated/
   :template: class.rst

   AnyClassifier
   EasyPreprocessor
   SimpleClassifier
   SimpleRegressor


Full API
=========

Datasets
--------

.. currentmodule:: dabl.datasets
.. autosummary::
   :toctree: generated/
   :template: function.rst

   load_adult
   load_ames
   load_titanic


Plotting
--------

.. currentmodule:: dabl.plot
.. autosummary::
    :toctree: generated/
    :template: function.rst

    class_hists
    discrete_scatter
    find_pretty_grid
    mosaic_plot
    plot_classification_categorical
    plot_classification_continuous
    plot_coefficients
    plot_regression_categorical
    plot_regression_continuous


Portfolios
-----------
Built-in lists of classifiers and regressors to search.

.. currentmodule:: dabl.pipelines
.. autosummary::
   :toctree: generated/
   :template: function.rst

   get_any_classifiers
   get_fast_classifiers
   get_fast_regressors
