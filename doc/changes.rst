Release History
===============

dabl 0.2.1 (current development version
---------------------------------------
- nothing yet

dabl 0.2.0
-----------
- Rely on the Successive Halving implementation from scikit-learn 0.24, removing the old implementation.
  Consequently the search module in dabl has been deprecated and the minimum version requirement of scikit-learn is now 0.24.

- The type detection has been completely rewritten and accomodates more edge cases, :issue:`270` by :user:`amueller`.

- A global configuration was introduced that can be set with `set_config`. For now, this allows users to turn off truncation of labels, by :user:`amueller`.
- Fix default value of `alpha` in `plot_regression_continuous`, :issue:`276` by :user:`amueller`.
- Fix a memory issue when calling bincount on really large integers in the type detection, :issue:`275` by :user:`amueller`.

dabl 0.1.9
-------------
- Fix bug in type detection when a column contained boolean data and missing values, :issue:`256` by :user:`amueller`.
- Bundle LICENSE file with project in release, :issue:`253` by :user:`dhirschfeld`.
- Make color usage consistent between scatter plots and mosaic plots, :issue:`249` by :user:`h4pZ`.
- Update the AnyClassifier portfolio to include several new optimized portfolios, :issue:`246` by :user:`hp2500`.


dabl  0.1.7
------------
- Ensure target column is not dropped in 'clean' for highly imbalanced datasets #171.
- Scale histograms separately in class histograms #173.
- Shorten really long column names to fix figure layout #180.
- Add shuffling to cross-validation for simple models #185.
- Fix broken legend for class histograms for ordinal variables #189.
- Allow numpy arrays in SimpleRegressor and plot #187.
- Add actual vs predicted plot for regression to explain #186.


dabl 0.1.6
-----------
- More fixed to dirty floats with heterogeneous dtypes.

dabl 0.1.5
----------
- More robust detection of dirty floats, more robust parsing of categorical variables.
- Ensure data is parsed consistently between predict and fit by not calling `clean` in fit.
- Allow passing columns with integer names as target in `plot`.
