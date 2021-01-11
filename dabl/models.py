import warnings
import numpy as np
import pandas as pd

import sklearn

from sklearn.metrics import make_scorer, average_precision_score
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import StratifiedKFold, KFold
try:
    from sklearn.metrics._scorer import _check_multimetric_scoring
except ImportError:
    from sklearn.metrics.scorer import _check_multimetric_scoring
from sklearn.model_selection._validation import _fit_and_score
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.metaestimators import if_delegate_has_method
try:
    from sklearn.utils._testing import set_random_state
except ImportError:
    from sklearn.utils.testing import set_random_state

from sklearn.dummy import DummyClassifier
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingGridSearchCV


from .preprocessing import EasyPreprocessor, detect_types
from .pipelines import (get_fast_classifiers, get_fast_regressors,
                        get_any_classifiers)
from .utils import nice_repr, _validate_Xyt


def _format_scores(scores):
    return " ".join(('{}: {:.3f}'.format(name, score)
                     for name, score in scores.items()))


class _DablBaseEstimator(BaseEstimator):

    @if_delegate_has_method(delegate='est_')
    def predict_proba(self, X):
        return self.est_.predict_proba(X)

    @if_delegate_has_method(delegate='est_')
    def decision_function(self, X):
        return self.est_.decision_function(X)


class _BaseSimpleEstimator(_DablBaseEstimator):
    def predict(self, X):
        if not self.refit:
            raise ValueError("Must specify refit=True to predict.")
        with warnings.catch_warnings():
            # fix when requiring sklearn 0.22
            # check_is_fitted will not have arguments any more
            warnings.filterwarnings('ignore', category=FutureWarning)
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            check_is_fitted(self, 'est_')

        return self.est_.predict(X)

    def _evaluate_one(self, estimator, data_preproc, scorers):
        res = []
        for X_train, X_test, y_train, y_test in data_preproc:
            X = np.vstack([X_train, X_test])
            if y_train.ndim < 2 and y_test.ndim < 2:
                y = np.hstack([y_train, y_test])
            else:
                y = np.vstack([y_train, y_test])
            train = np.arange(len(X_train))
            test = np.arange(len(X_train), len(X_test) + len(X_train))
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',
                                        category=UndefinedMetricWarning)
                scores = _fit_and_score(estimator, X, y, scorer=scorers,
                                        train=train, test=test,
                                        parameters={}, fit_params={},
                                        verbose=self.verbose)
            res.append(scores['test_scores'])

        res_mean = pd.DataFrame(res).mean(axis=0)
        try:
            # show only last step of pipeline for simplicity
            name = nice_repr(estimator.steps[-1][1])
        except AttributeError:
            name = nice_repr(estimator)

        if self.verbose:
            print("Running {}".format(name))
            print(_format_scores(res_mean))
        res_mean.name = name
        self.log_.append(res_mean)
        return res_mean

    def _fit(self, X, y=None, target_col=None):
        """Fit estimator.

        Requires to either specify the target as separate 1d array or Series y
        (in scikit-learn fashion) or as column of the dataframe X specified by
        target_col.
        If y is specified, X is assumed not to contain the target.

        Parameters
        ----------
        X : DataFrame
            Input features. If target_col is specified, X also includes the
            target.
        y : Series or numpy array, optional.
            Target. You need to specify either y or target_col.
        target_col : string or int, optional
            Column name of target if included in X.
        """
        X, y = _validate_Xyt(X, y, target_col, do_clean=False)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        types = detect_types(X, type_hints=self.type_hints)
        self.feature_names_ = X.columns
        self.types_ = types

        y, self.scoring_ = self._preprocess_target(y)
        self.log_ = []

        # reimplement cross-validation so we only do preprocessing once
        # This could/should be solved with dask?
        if isinstance(self, RegressorMixin):
            # this is how inheritance works, right?
            cv = KFold(n_splits=5, shuffle=self.shuffle,
                       random_state=self.random_state)
        elif isinstance(self, ClassifierMixin):
            cv = StratifiedKFold(
                n_splits=5, shuffle=self.shuffle,
                random_state=self.random_state)
        data_preproc = []
        for i, (train, test) in enumerate(cv.split(X, y)):
            # maybe do two levels of preprocessing
            # to search over treatment of categorical variables etc
            # Also filter?
            verbose = self.verbose if i == 0 else 0
            sp = EasyPreprocessor(verbose=verbose, types=types)
            X_train = sp.fit_transform(X.iloc[train], y.iloc[train])
            X_test = sp.transform(X.iloc[test])
            data_preproc.append((X_train, X_test, y.iloc[train], y.iloc[test]))

        estimators = self._get_estimators()
        rank_scoring = self._rank_scoring
        self.current_best_ = {rank_scoring: -np.inf}
        for est in estimators:
            set_random_state(est, self.random_state)
            scorers = _check_multimetric_scoring(est, self.scoring_)
            scores = self._evaluate_one(est, data_preproc, scorers)
            # make scoring configurable
            if scores[rank_scoring] > self.current_best_[rank_scoring]:
                if self.verbose:
                    print("=== new best {} (using {}):".format(
                        scores.name,
                        rank_scoring))
                    print(_format_scores(scores))
                    print()

                self.current_best_ = scores
                best_est = est
        if self.verbose:
            print("\nBest model:\n{}\nBest Scores:\n{}".format(
                  nice_repr(best_est), _format_scores(self.current_best_)))
        if self.refit:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                self.est_ = make_pipeline(EasyPreprocessor(types=types),
                                          best_est)
                self.est_.fit(X, y)
        return self


class SimpleClassifier(_BaseSimpleEstimator, ClassifierMixin):
    """Automagic anytime classifier.

    Parameters
    ----------
    refit : boolean, True
        Whether to refit the model on the full dataset.

    random_state : random state, int or None (default=None)
        Random state or seed.

    verbose : integer, default=1
        Verbosity (higher is more output).

    type_hints : dict or None
            If dict, provide type information for columns.
            Keys are column names, values are types as provided by
            detect_types.

    shuffle : boolean, default=True
        Whether to shuffle the training set in cross-validation.

    Attributes
    ----------
    est_ : sklearn estimator
        Best estimator found.

    """
    def __init__(self, refit=True, random_state=None, verbose=1,
                 type_hints=None, shuffle=True):
        self.verbose = verbose
        self.random_state = random_state
        self.refit = refit
        self.type_hints = type_hints
        self.shuffle = shuffle

    def _get_estimators(self):
        return get_fast_classifiers(n_classes=len(self.classes_))

    def _preprocess_target(self, y):
        target_type = type_of_target(y)
        le = LabelEncoder().fit(y)
        y = pd.Series(y)
        self.classes_ = le.classes_

        if target_type == "binary":
            minority_class = y.value_counts().index[1]
            my_average_precision_scorer = make_scorer(
                average_precision_score, pos_label=minority_class,
                needs_threshold=True)
            scoring = {'accuracy': 'accuracy',
                       'average_precision': my_average_precision_scorer,
                       'roc_auc': 'roc_auc',
                       'recall_macro': 'recall_macro',
                       'f1_macro': 'f1_macro'
                       }
        elif target_type == "multiclass":
            scoring = ['accuracy', 'recall_macro', 'precision_macro',
                       'f1_macro']
        else:
            raise ValueError("Unknown target type: {}".format(target_type))
        return y, scoring

    def fit(self, X, y=None, *, target_col=None):
        """Fit classifier.

        Requires to either specify the target as separate 1d array or Series y
        (in scikit-learn fashion) or as column of the dataframe X specified by
        target_col.
        If y is specified, X is assumed not to contain the target.

        Parameters
        ----------
        X : DataFrame
            Input features. If target_col is specified, X also includes the
            target.
        y : Series or numpy array, optional.
            Target class labels. You need to specify either y or target_col.
        target_col : string or int, optional
            Column name of target if included in X.
        """
        self._rank_scoring = "recall_macro"
        return self._fit(X=X, y=y, target_col=target_col)


class SimpleRegressor(_BaseSimpleEstimator, RegressorMixin):
    """Automagic anytime classifier.

    Parameters
    ----------
    refit : boolean, True
        Whether to refit the model on the full dataset (I think).

    random_state : random state, int or None (default=None)
        Random state or seed.

    verbose : integer, default=1
        Verbosity (higher is more output).

    type_hints : dict or None
            If dict, provide type information for columns.
            Keys are column names, values are types as provided by
            detect_types.

    shuffle : boolean, default=True
        Whether to shuffle the training set in cross-validation.
    """
    def __init__(self, refit=True, random_state=None, verbose=1,
                 type_hints=None, shuffle=True):
        self.verbose = verbose
        self.refit = refit
        self.random_state = random_state
        self.type_hints = type_hints
        self.shuffle = shuffle

    def _get_estimators(self):
        return get_fast_regressors()

    def _preprocess_target(self, y):
        target_type = type_of_target(y)

        if target_type not in ["continuous", "multiclass"]:
            # if all labels are integers type_of_target is multiclass.
            # We trust our user they mean regression.
            raise ValueError("Unknown target type: {}".format(target_type))
        scoring = ('r2', 'neg_mean_squared_error')
        return y, scoring

    def fit(self, X, y=None, *, target_col=None):
        """Fit regressor.

        Requires to either specify the target as separate 1d array or Series y
        (in scikit-learn fashion) or as column of the dataframe X specified by
        target_col.
        If y is specified, X is assumed not to contain the target.

        Parameters
        ----------
        X : DataFrame
            Input features. If target_col is specified, X also includes the
            target.
        y : Series or numpy array, optional.
            Target class labels. You need to specify either y or target_col.
        target_col : string or int, optional
            Column name of target if included in X.

        """
        self._rank_scoring = "r2"

        return self._fit(X=X, y=y, target_col=target_col)


class AnyClassifier(_DablBaseEstimator, ClassifierMixin):
    """Classifier with automatic model selection.

    This model uses successive halving on a portfolio of complex models
    (HistGradientBoosting, RandomForest, SVC, LogisticRegression)
    to pick the best model family and hyper-parameters.

    AnyClassifier internally applies EasyPreprocessor, so no preprocessing
    is necessary.

    Parameters
    ----------
    n_jobs : int, default=None
        Number of processes to spawn for parallelizing the search.

    min_resources : {‘exhaust’, ‘smallest’} or int, default=’exhaust’
        The minimum amount of resource that any candidate is allowed to use
        for a given iteration.  Equivalently, this defines the amount of
        resources r0 that are allocated for each candidate at the first
        iteration. See the documentation of HalvingGridSearchCV for more
        information.

    verbose : integer, default=0
        Verbosity. Higher means more output.

    type_hints : dict or None
            If dict, provide type information for columns.
            Keys are column names, values are types as provided by
            detect_types.

     portfolio : str, default='baseline'
             Lets you choose a portfolio. Choose 'baseline' for multiple
             classifiers with default parameters, 'hgb' for
             high-performing HistGradientBoostingClassifiers,
             'svc' for high-performing support vector classifiers,
             'rf' for high-performing random forest classifiers,
             'lr' for high-performing logistic regression classifiers,
             'mixed' for a portfolio of different high-performing
             classifiers.

    Attributes
    ----------
    search_ : HalvingGridSearchCV instance
        Fitted HalvingGridSearchCV instance for inspection.

    est_ : sklearn estimator
        Best estimator (pipeline) found during search.

    """
    def __init__(self, n_jobs=None, min_resources='exhaust', verbose=0,
                 type_hints=None, portfolio='baseline'):
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.min_resources = min_resources
        self.type_hints = type_hints
        self.portfolio = portfolio

    def _get_estimators(self):
        return get_any_classifiers(portfolio=self.portfolio)

    def _preprocess_target(self, y):
        # copy and paste from above, should be a mixin
        target_type = type_of_target(y)
        le = LabelEncoder().fit(y)
        y = pd.Series(y)
        self.classes_ = le.classes_

        if target_type == "binary":
            scoring = 'recall_macro'
        elif target_type == "multiclass":
            scoring = 'recall_macro'
        else:
            raise ValueError("Unknown target type: {}".format(target_type))
        return y, scoring

    def predict(self, X):
        with warnings.catch_warnings():
            # fix when requiring sklearn 0.22
            # check_is_fitted will not have arguments any more
            warnings.filterwarnings('ignore', category=FutureWarning)
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            check_is_fitted(self, 'est_')

        return self.est_.predict(X)

    def fit(self, X, y=None, *, target_col=None):
        """Fit estimator.

        Requires to either specify the target as separate 1d array or Series y
        (in scikit-learn fashion) or as column of the dataframe X specified by
        target_col.
        If y is specified, X is assumed not to contain the target.

        Parameters
        ----------
        X : DataFrame
            Input features. If target_col is specified, X also includes the
            target.
        y : Series or numpy array, optional.
            Target. You need to specify either y or target_col.
        target_col : string or int, optional
            Column name of target if included in X.
        """
        # copy and paste from above?!
        if ((y is None and target_col is None)
                or (y is not None) and (target_col is not None)):
            raise ValueError(
                "Need to specify either y or target_col.")
        X, y = _validate_Xyt(X, y, target_col, do_clean=False)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        types = detect_types(X, type_hints=self.type_hints)
        self.feature_names_ = X.columns
        self.types_ = types
        cv = 5
        factor = 3

        y, self.scoring_ = self._preprocess_target(y)
        self.log_ = []

        # reimplement cross-validation so we only do preprocessing once
        pipe = Pipeline([('preprocessing',
                          EasyPreprocessor(verbose=self.verbose, types=types)),
                         ('classifier', DummyClassifier())])

        estimators = self._get_estimators()
        param_grid = [{'classifier': [est]} for est in estimators]
        gs = HalvingGridSearchCV(
            factor=factor,
            estimator=pipe, param_grid=param_grid,
            min_resources=self.min_resources,
            verbose=self.verbose, cv=cv, error_score='raise',
            scoring=self.scoring_, refit='recall_macro', n_jobs=self.n_jobs)
        self.search_ = gs
        with sklearn.config_context(print_changed_only=True):
            gs.fit(X, y)
        self.est_ = gs.best_estimator_

        print("best classifier: ", gs.best_params_['classifier'])
        print("best score: {:.3f}".format(gs.best_score_))

        return self
