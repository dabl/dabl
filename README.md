# dabl
The data analysis baseline library.

- "Mr Sanchez, are you a data scientist?"
- "I dabl, Mr president."

Find more information on the [website](https://dabl.github.io/).

## State of the library
Right now, this library is still a prototype. API might change, and you shouldn't rely on it in any critical settings.

## Try it out

```
pip install dabl
```

or [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dabl/dabl/master)

## Current scope and upcoming features
This library is very much still under development. Current code focuses mostly on exploratory visualization and preprocessing.
There are also drop-in replacements for GridSearchCV and RandomizedSearchCV using successive halfing.
There are preliminary portfolios in the style of
[POSH
auto-sklearn](https://ml.informatik.uni-freiburg.de/papers/18-AUTOML-AutoChallenge.pdf)
to find strong models quickly.  In essence that boils down to a quick search
over different gradient boosting models and other tree ensembles and
potentially kernel methods.

Stay Tuned!

## Pandas Profiling package
The Pandas Profiling package is useful for initial data analysis. Using Pandas Profiling can provide a thorough summary of the data in only a single line of code. Using the ```ProfileReport()``` method, you are able to access a HTML report of your data that can help you find correlations and identify missing data.

## Try it out

```
pip install pandas-profiling
```

or [https://github.com/pandas-profiling/pandas-profiling]
