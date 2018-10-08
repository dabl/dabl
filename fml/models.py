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

from .preprocessing import FriendlyPreprocessor


class FriendlyClassifier(BaseEstimator, ClassifierMixin):
    """Automagic anytime classifier
    """
    def __init__(self, memory=None):
        self.memory = memory

    def fit(self, X, y):

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
                             'precision_macro': 'precision_macro'
                             }
        elif target_type == "multiclass":
            self.scoring_ = ['accuracy', 'precision_macro',
                             'recall_macro']
        else:
            raise ValueError("Unknown target type: {}".format(target_type))
        # speed up label encoding by not redoing it
        self.log_ = []
        n_classes = len(np.unique(y))
        # kwargs = {'memory': self.memory}
        # Heuristic: start with fast / instantaneous models
        cv = StratifiedKFold(n_splits=5)
        data_preproc = []
        for train, test in cv.split(X, y):
            sp = FriendlyPreprocessor()
            X_train = sp.fit_transform(X.iloc[train], y.iloc[train])
            X_test = sp.transform(X.iloc[test])
            data_preproc.append((X_train, X_test, y.iloc[train], y.iloc[test]))

        fast_ests = [DummyClassifier(strategy="prior"),
                     GaussianNB(),
                     make_pipeline(MinMaxScaler(), MultinomialNB()),
                     DecisionTreeClassifier(max_depth=1,
                                            class_weight="balanced"),
                     DecisionTreeClassifier(max_depth=max(5, n_classes),
                                            class_weight="balanced"),
                     ]

        scorers, _ = _check_multimetric_scoring(fast_ests[1],
                                                scoring=self.scoring_)

        self.current_best_ = -np.inf
        for ests in fast_ests:
            self._evaluate_one(ests, data_preproc, scorers)

    def predict(self, X):
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
            name = str(estimator.steps[-1][1])
        except AttributeError:
            name = str(estimator)
        # FIXME don't use accuracy for getting the best?
        print(name)
        print(res_mean)
        res_mean.name = name
        self.log_.append(res_mean)
