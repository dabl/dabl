# dabl
The data analysis baseline library.

- "Mr Sanchez, are you a data scientist?"
- "I dabl, Mr president."

## Warning
This is pre-alpha software and is still very-much in flux.

*Dependency*: `dabl` uses some features from the development version of `sklearn`, which will eventually be a part of the 0.21 release.

## Current scope and upcoming features
This library is very much still under development. Current code focusses mostly on exploratory visualiation and preprocessing.
There are also drop-in replacements for GridSearchCV and RandomizedSearchCV using successive halfing.
The next step in the development will be adding portfolios in the style of
[POSH
auto-sklearn](https://ml.informatik.uni-freiburg.de/papers/18-AUTOML-AutoChallenge.pdf)
to find strong models quickly.  In essence that boils down to a quick search
over different gradient boosting models and other tree ensembles and
potentially kernel methods.

Stay Tuned!
