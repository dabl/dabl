Welcome to dabl, the Data Analysis Baseline Library
===================================================

This project tries to help make supervised machine learning more accessible for
beginners, and reduce boiler plate for common tasks.

This library is in very active development, so it's not recommended for production use.

Development at `github.com/amueller/dabl <https://github.com/amueller/dabl>`_.

Examples
--------
A minimum example of using dabl for classification is:

    >>> import dabl
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.datasets import load_digits
    >>> X, y = load_digits(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    >>> sc = dabl.SimpleClassifier().fit(X_train, y_train)
    Running ...
    >>> print("Accuracy score", sc.score(X_test, y_test))
    Accuracy score 0.98


This will return increasingly better results immediately and should conclude
within several seconds with an accuracy of 0.98.


The real strength of ``dabl`` is in providing simple interfaces for data exploration.
Here are some examples of visualizations produced simply by calling ``plot(data, 'target_col')``:

.. figure:: ../auto_examples/images/plot/sphx_glr_ames_003.png
    :target: ../auto_examples/plot/plot_ames.html
    :align: center
    :scale: 50
 
    Impact of categorical variables in the ames housing regression dataset.

  

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   quick_start

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   user_guide
   api

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Examples

   auto_examples/index

`Getting started <quick_start.html>`_
-------------------------------------

A quick guide on how to use dabl tools in practice.

`User Guide <user_guide.html>`_
-------------------------------

A full guide to the main concepts and ideas of the dabl.

`API Documentation <api.html>`_
-------------------------------

A documentation of all the dabl classes and functions.

`Examples <auto_examples/index.html>`_
--------------------------------------

Some examples to give you a taste how much you can achieve with little code
with some help from dabl!
