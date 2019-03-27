import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import make_scorer, average_precision_score
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.scorer import _check_multimetric_scoring
from sklearn.model_selection._validation import _multimetric_score
from sklearn.linear_model import LogisticRegression

from .preprocessing import FriendlyPreprocessor
from .utils import nice_repr


class FriendlyClassifier(BaseEstimator, ClassifierMixin):
    """Automagic anytime classifier.

    Parameters
    ----------
    refit : boolean, False
        Whether to refit the model on the full dataset (I think).

    verbose : integer, default=1
        Verbosity (higher is more output)
    """
    def __init__(self, refit=False, verbose=1):
        self.verbose = verbose
        self.refit = refit

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

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
        n_classes = len(np.unique(y))

        # reimplement cross-validation so we only do preprocessing once
        # This could/should be solved with dask?
        cv = StratifiedKFold(n_splits=5)
        data_preproc = []
        for i, (train, test) in enumerate(cv.split(X, y)):
            # maybe do two levels of preprocessing
            # to search over treatment of categorical variables etc
            # Also filter?
            verbose = self.verbose if i == 0 else 0
            sp = FriendlyPreprocessor(verbose=verbose)
            X_train = sp.fit_transform(X.iloc[train], y.iloc[train])
            X_test = sp.transform(X.iloc[test])
            data_preproc.append((X_train, X_test, y.iloc[train], y.iloc[test]))
        # Heuristic: start with fast / instantaneous models

        fast_ests = [DummyClassifier(strategy="prior"),
                     GaussianNB(),
                     make_pipeline(MinMaxScaler(), MultinomialNB()),
                     DecisionTreeClassifier(max_depth=1,
                                            class_weight="balanced"),
                     DecisionTreeClassifier(max_depth=max(5, n_classes),
                                            class_weight="balanced"),
                     DecisionTreeClassifier(class_weight="balanced",
                                            min_impurity_decrease=.01),
                     LogisticRegression(C=.1, solver='lbfgs',
                                        class_weight='balanced')
                     ]

        scorers, _ = _check_multimetric_scoring(fast_ests[1],
                                                scoring=self.scoring_)

        self.current_best_ = -np.inf
        for est in fast_ests:
            scores = self._evaluate_one(est, data_preproc, scorers)
            # make scoring configurable
            this_score = scores['recall_macro']
            if this_score > self.current_best_:
                if self.verbose:
                    print("new best: {:.4f}".format(this_score))
                self.current_best_ = this_score
                best_est = est
        if self.refit:
            self.est_ = make_pipeline(FriendlyPreprocessor(), best_est)
            self.est_.fit(X, y)

    def predict(self, X):
        # FIXME check self.refit
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
            res_string = "".join("{}: {:.4f}    ".format(m, s)
                                 for m, s in res_mean.items())
            print(res_string)
        res_mean.name = name
        self.log_.append(res_mean)
        return res_mean
