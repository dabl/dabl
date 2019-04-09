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
   plot_supervised

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SimpleClassifier
   EasyPreprocessor


Supervised Models
=================

.. autosummary::
    :toctree: generated/
    :template: class.rst

   SimpleClassifier

Preprocessing
=============

.. currentmodule:: dabl.preprocessing

.. autosummary::
    :toctree: generated/
    :template: class.rst

    EasyPreprocessor

.. autosummary::
    :toctree: generated/
    :template: function.rst

    detect_types
    clean

Plotting
=========

.. currentmodule:: dabl.plot
.. autosummary::
    :toctree: generated/
    :template: function.rst

    class_hists
    discrete_scatter
    find_pretty_grid
    mosaic_plot
    plot_supervised
    plot_classification_categorical
    plot_classification_continuous
    plot_regression_categorical
    plot_regression_continuous

Model Search
=============

.. currentmodule:: dabl.search
.. autosummary::
   :toctree: generated/
   :template: function.rst

   GridSuccessiveHalving
   RandomSuccessiveHalving


Portfolios
==========

.. currentmodule:: dabl.pipelines
.. autosummary::
   :toctree: generated/
   :template: function.rst

   get_fast_classifiers


Datasets
==========

.. currentmodule:: dabl.datasets
.. autosummary::
   :toctree: generated/
   :template: function.rst

   load_ames
   load_titanic
