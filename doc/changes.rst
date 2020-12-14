Release History
===============

dabl 0.2.0 (current development)
--------------------------------
- Rely on the Successive Halving implementation from scikit-learn 0.24, removing the old implementation.
  Consequently the search module in dabl has been deprecated and the minimum version requirement of scikit-learn is now 0.24.
 
dabl 0.1.9
-------------
- Fix bug in type detection when a column contained boolean data and missing values, :issue:`256` by :user:`amueller`.
- Bundle LICENSE file with project in release, :issue:`253` by :user:`dhirschfeld`.
- Make color usage consistent between scatter plots and mosaic plots, :issue:`249` by :user:`h4pZ`.
- Update the AnyClassifier portfolio to include several new optimized portfolios, :issue:`246` by :user:`hp2500`.
