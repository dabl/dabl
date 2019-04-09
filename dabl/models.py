import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import make_scorer, average_precision_score
from sklearn.base import BaseEstimator, ClassifierMixin, clone, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics.scorer import _check_multimetric_scoring
from sklearn.model_selection._validation import _multimetric_score
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.testing import set_random_state
from sklearn.dummy import DummyClassifier


from .preprocessing import EasyPreprocessor, clean, detect_types
from .pipelines import (get_fast_classifiers, get_fast_regressors,
                        get_any_classifiers)
from .utils import nice_repr
from .search import GridSuccessiveHalving


class _BaseSimpleEstimator(BaseEstimator):

    def predict(self, X):
        if not self.refit:
            raise ValueError("Must specify refit=True to predict.")
        check_is_fitted(self, 'est_')
        self.est_.predict(X)

    def _evaluate_one(self, estimator, data_preproc, scorers):
        res = []
        for X_train, X_test, y_train, y_test in data_preproc:
            est = clone(estimator)
            est.fit(X_train, y_train)
            # fit_time = time.time() - start_time
            # _score will return dict if is_multimetric is True
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',
                                        category=UndefinedMetricWarning)
                test_scores = _multimetric_score(est, X_test, y_test, scorers)
            # score_time = time.time() - start_time - fit_time
            # train_scores = _multimetric_score(estimator, X_train, y_train,
            #                                   scorers)
            res.append(test_scores)

        res_mean = pd.DataFrame(res).mean(axis=0)
        try:
            # show only last step of pipeline for simplicity
            name = nice_repr(estimator.steps[-1][1])
        except AttributeError:
            name = nice_repr(estimator)

        if self.verbose:
            print(name)
            res_string = "".join("{}: {:.3f}    ".format(m, s)
                                 for m, s in res_mean.items())
            print(res_string)
        res_mean.name = name
        self.log_.append(res_mean)
        return res_mean

    def _fit(self, X, y=None, target_col=None):
        """Fit estimator.

        Requiers to either specify the target as separate 1d array or Series y
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
        if ((y is None and target_col is None)
                or (y is not None) and (target_col is not None)):
            raise ValueError(
                "Need to specify exactly one of y and target_col.")
        X = clean(X)
        if target_col is not None:
            y = X[target_col]
            X = X.drop(target_col, axis=1)
        types = detect_types(X)
        self.feature_names_ = X.columns
        self.types_ = types

        y, self.scoring_ = self._preprocess_target(y)
        self.log_ = []

        # reimplement cross-validation so we only do preprocessing once
        # This could/should be solved with dask?
        if isinstance(self, RegressorMixin):
            # this is how inheritance works, right?
            cv = KFold(n_splits=5)
        elif isinstance(self, ClassifierMixin):
            cv = StratifiedKFold(n_splits=5)
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
        scorers, _ = _check_multimetric_scoring(estimators[1],
                                                scoring=self.scoring_)
        rank_scoring = self._rank_scoring
        self.current_best_ = {rank_scoring: -np.inf}
        for est in estimators:
            set_random_state(est, self.random_state)
            scores = self._evaluate_one(est, data_preproc, scorers)
            # make scoring configurable
            if scores[rank_scoring] > self.current_best_[rank_scoring]:
                if self.verbose:
                    with pd.option_context('precision', 3):
                        print("new best (using {}):\n{}".format(
                            rank_scoring, scores))
                self.current_best_ = scores
                best_est = est
        if self.verbose:
            with pd.option_context('precision', 3):
                print("Best model:\n{}\nBest Scores:\n{}".format(
                      nice_repr(best_est), self.current_best_))
        if self.refit:
            self.est_ = make_pipeline(EasyPreprocessor(), best_est)
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
        Verbosity (higher is more output)
    """
    def __init__(self, refit=True, random_state=None, verbose=1):
        self.verbose = verbose
        self.random_state = random_state
        self.refit = refit

    def _get_estimators(self):
        return get_fast_classifiers(n_classes=len(self.classes_))

    def _preprocess_target(self, y):
        target_type = type_of_target(y)
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y))
        self.classes_ = le.classes_

        if target_type == "binary":
            minority_class = y.value_counts().index[1]
            my_average_precision_scorer = make_scorer(
                average_precision_score, pos_label=minority_class,
                needs_threshold=True)
            scoring = {'accuracy': 'accuracy',
                       'average_precision': my_average_precision_scorer,
                       'roc_auc': 'roc_auc',
                       'recall_macro': 'recall_macro'
                       }
        elif target_type == "multiclass":
            scoring = ['accuracy', 'recall_macro', 'precision_macro']
        else:
            raise ValueError("Unknown target type: {}".format(target_type))
        return y, scoring

    def fit(self, X, y=None, target_col=None):
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
        Verbosity (higher is more output)
    """
    def __init__(self, refit=True, random_state=None, verbose=1):
        self.verbose = verbose
        self.refit = refit
        self.random_state = random_state

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

    def fit(self, X, y=None, target_col=None):
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


class AnyClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_jobs=None, force_exhaust_budget=False, verbose=0):
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.force_exhaust_budget = force_exhaust_budget

    def _get_estimators(self):
        return get_any_classifiers()

    def _preprocess_target(self, y):
        # copy and paste from above, should be a mixin
        target_type = type_of_target(y)
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y))
        self.classes_ = le.classes_

        if target_type == "binary":
            minority_class = y.value_counts().index[1]
            my_average_precision_scorer = make_scorer(
                average_precision_score, pos_label=minority_class,
                needs_threshold=True)
            scoring = {'accuracy': 'accuracy',
                       'average_precision': my_average_precision_scorer,
                       'roc_auc': 'roc_auc',
                       'recall_macro': 'recall_macro'
                       }
        elif target_type == "multiclass":
            scoring = ['accuracy', 'recall_macro', 'precision_macro']
        else:
            raise ValueError("Unknown target type: {}".format(target_type))
        return y, scoring

    def fit(self, X, y=None, target_col=None):
        """Fit estimator.

        Requiers to either specify the target as separate 1d array or Series y
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
                "Need to specify exactly one of y and target_col.")
        X = clean(X)
        if target_col is not None:
            y = X[target_col]
            X = X.drop(target_col, axis=1)
        types = detect_types(X)
        self.feature_names_ = X.columns
        self.types_ = types

        y, self.scoring_ = self._preprocess_target(y)
        self.log_ = []

        # reimplement cross-validation so we only do preprocessing once
        pipe = Pipeline([('preprocessing',
                          EasyPreprocessor(verbose=self.verbose, types=types)),
                         ('classifier', DummyClassifier())])

        estimators = self._get_estimators()
        param_grid = [{'classifier': [est]} for est in estimators]
        gs = GridSuccessiveHalving(
            estimator=pipe, param_grid=param_grid,
            force_exhaust_budget=self.force_exhaust_budget,
            verbose=self.verbose, refit=False, cv=5, error_score='raise',
            scoring='recall_macro')
        self.search_ = gs
        gs.fit(X, y)
        print("best classifier: ", gs.best_params_['classifier'])
        print("best score: {:.3f}".format(gs.best_score_))
        return self
