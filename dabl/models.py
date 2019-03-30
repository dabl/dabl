import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import make_scorer, average_precision_score
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.scorer import _check_multimetric_scoring
from sklearn.model_selection._validation import _multimetric_score
from sklearn.utils.validation import check_is_fitted

from .preprocessing import EasyPreprocessor, clean, detect_types
from .pipelines import get_fast_classifiers
from .utils import nice_repr


class EasyClassifier(BaseEstimator, ClassifierMixin):
    """Automagic anytime classifier.

    Parameters
    ----------
    refit : boolean, True
        Whether to refit the model on the full dataset (I think).

    verbose : integer, default=1
        Verbosity (higher is more output)
    """
    def __init__(self, refit=True, verbose=1):
        self.verbose = verbose
        self.refit = refit

    def fit(self, X, y=None, target_col=None):
        """Fit classifier.

        Requiers to either specify the target as seperate 1d array or Series y
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
        # fixme store types!? call clean with type hints?
        target_type = type_of_target(y)
        y = pd.Series(LabelEncoder().fit_transform(y))

        if target_type == "binary":
            minority_class = y.value_counts().index[1]
            my_average_precision_scorer = make_scorer(
                average_precision_score, pos_label=minority_class,
                needs_threshold=True)
            self.scoring_ = {'accuracy': 'accuracy',
                             'average_precision': my_average_precision_scorer,
                             'roc_auc': 'roc_auc',
                             'recall_macro': 'recall_macro'
                             }
        elif target_type == "multiclass":
            self.scoring_ = ['accuracy', 'recall_macro', 'precision_macro']
        else:
            raise ValueError("Unknown target type: {}".format(target_type))
        # speed up label encoding by not redoing it
        self.log_ = []

        # reimplement cross-validation so we only do preprocessing once
        # This could/should be solved with dask?
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

        # Heuristic: start with fast / instantaneous models
        fast_classifiers = get_fast_classifiers(n_classes=y.nunique())
        scorers, _ = _check_multimetric_scoring(fast_classifiers[1],
                                                scoring=self.scoring_)

        self.current_best_ = {'recall_macro': -np.inf}
        for est in fast_classifiers:
            scores = self._evaluate_one(est, data_preproc, scorers)
            # make scoring configurable
            if scores['recall_macro'] > self.current_best_['recall_macro']:
                if self.verbose:
                    print("new best (using recall macro):\n{}".format(
                        scores))
                self.current_best_ = scores
                best_est = est
        if self.verbose:
            print("Best model:\n{}\nBest Scores:\n{}".format(
                nice_repr(best_est), self.current_best_))
        if self.refit:
            self.est_ = make_pipeline(EasyPreprocessor(), best_est)
            self.est_.fit(X, y)
        return self

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
        name = name.replace("\s", " ")
        if self.verbose:
            print(name)
            res_string = "".join("{}: {:.4f}    ".format(m, s)
                                 for m, s in res_mean.items())
            print(res_string)
        res_mean.name = name
        self.log_.append(res_mean)
        return res_mean
